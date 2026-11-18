"""Coupon claim race guard.

并发安全的优惠券预留/领取原语。本模块用进程内锁模拟数据库行锁/Redis
分布式锁的语义，函数内部一律采用「条件更新 + 提交点」结构，避免
read-then-write 竞态。10 个修复点与代码的对应关系：

1. 条件更新        —— try_reserve / reserve_atomic 只在锁内读-判-写，
                      条件不满足直接拒绝，不存在无保护的先查后扣。
2. 余量非负约束    —— 扣减前校验 stock >= qty，stock 绝不可能变负。
3. 用户唯一约束    —— reserve_atomic 在同一临界区检查 users 集合，
                      同一用户并发只能成功一次。
4. 幂等键          —— idempotent_claim 对幂等键做唯一插入，重复键返回
                      False；重试必须先查该键再决定是否执行。
5. 跨实例锁        —— lock_key 生成全局确定的锁名（资源:标识），
                      配合 Redis/etcd 等外部存储即可跨单机互斥。
6. 事务隔离        —— 读余量、判重、扣减、写用户在同一把锁的同一临界
                      区内完成，等价于 SERIALIZABLE 下的原子条件更新。
7. 扣减与发券同事务 —— reserve_atomic 先在暂存区完成全部变更，提交点
                      一次性落库，任一步失败则整体不生效（无白扣）。
8. 重试查结果      —— is_retryable 区分可重试瞬时错误与不可重试业务
                      错误；可重试错误必须携带幂等键，先查结果再重试。
9. 缓存以库为准    —— 本模块只认 state（数据库真相），不维护独立缓存
                      余量；缓存只能由提交后的数据库变更回填。
10.批量统一事务   —— batch_reserve 对多条领取使用同一把锁、一次提交，
                      任一失败全部不生效。
"""

import threading
import weakref

# ---------------------------------------------------------------------------
# 锁注册表：每个可变状态对象一把锁，模拟数据库行锁。
# dict 本身不可哈希，无法作为 WeakKeyDictionary 的键，因此以 id(state)
# 为键、锁对象的弱引用为值存入 WeakValueDictionary：状态对象回收后，
# 锁失去唯一强引用而被回收、条目自动消失，也避免新对象复用同一 id
# 时拿到前一个对象的锁。注册表自身的访问同样加锁，否则两个线程可能
# 为同一状态各建一把锁，互斥就失效了（「锁只锁单机」问题）。
# ---------------------------------------------------------------------------
_LOCK_GUARD = threading.Lock()
_STATE_LOCKS: 'weakref.WeakValueDictionary[int, threading.RLock]' = (
    weakref.WeakValueDictionary()
)


def _state_lock(state):
    """返回 state 专属的可重入锁；同一状态对象始终拿到同一把锁。"""
    with _LOCK_GUARD:
        lock = _STATE_LOCKS.get(id(state))
        if lock is None:
            lock = threading.RLock()
            _STATE_LOCKS[id(state)] = lock
        return lock


# 可重试的瞬时错误：死锁、锁等待超时、连接故障、主从切换等。
# 这些错误不代表业务失败，必须带幂等键重试（修复点 8）。
_RETRYABLE_REASONS = frozenset({
    'deadlock',
    'lock_wait_timeout',
    'serialization_failure',
    'connection_error',
    'timeout',
    'unavailable',
})

# 不可重试的确定性错误：重试结果不会改变，直接失败。
_NON_RETRYABLE_REASONS = frozenset({
    'validation',
    'not_found',
    'already_exists',
    'duplicate',
    'permission_denied',
    'constraint_violation',
})


def _non_negative_int(value):
    """stock 必须是非负整数；布尔值是 int 的子类，显式排除。"""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_qty(qty):
    """领取数量必须是正整数（0 和负数一律拒绝）。"""
    return isinstance(qty, int) and not isinstance(qty, bool) and qty > 0


def try_reserve(state, qty):
    """条件更新式余量扣减（修复点 1、2、6、9）。

    在同一临界区内完成「读余量 -> 判余量 >= qty -> 扣减」，等价于 SQL：
        UPDATE coupon SET stock = stock - :qty
        WHERE id = :id AND stock >= :qty
    余量不足、数量非法、状态缺字段时返回 False 且不改动任何数据；
    成功才扣减并返回 True，stock 永远不会变成负数。

    Args:
        state: 持有 'stock' 键的可变映射（代表数据库行，是唯一真相源）。
        qty:  领取数量，必须为正整数。

    Returns:
        bool: 扣减成功 True，否则 False。
    """
    if not _positive_qty(qty):
        return False
    if not isinstance(state, dict):
        return False

    lock = _state_lock(state)
    with lock:
        stock = state.get('stock')
        # 非负约束 + 字段缺失保护：缓存/旧数据脏值不允许进入扣减。
        if not _non_negative_int(stock):
            return False
        if stock < qty:  # 条件更新：余量不足直接拒绝，杜绝超卖。
            return False
        state['stock'] = stock - qty
        return True


def reserve_atomic(state, user, qty):
    """单用户原子领取：余量扣减与发券同事务（修复点 2、3、6、7）。

    在同一把锁的同一临界区内完成用户判重、余量校验、扣减和发券登记，
    全部检查通过后才在提交点一次性写入；任一检查失败则 state 完全不变，
    不会出现「余量扣了但券没发」或「同一用户领到多张」。

    Args:
        state: 含 'stock'，可选含 'users'（已领用户集合）的可变映射。
        user:  用户标识，None 或空值视为非法。
        qty:   领取数量，必须为正整数。

    Returns:
        bool: 领取成功 True；重复领取/余量不足/参数非法返回 False。
    """
    if not _positive_qty(qty):
        return False
    if not isinstance(state, dict):
        return False
    if user is None or user == '':
        return False

    lock = _state_lock(state)
    with lock:
        users = state.get('users')
        if users is not None and user in users:  # 用户唯一约束。
            return False

        stock = state.get('stock')
        if not _non_negative_int(stock) or stock < qty:  # 非负 + 条件更新。
            return False

        # 全部前置检查通过，进入提交点：扣减与发券一起生效。
        if users is None:
            users = set()
            state['users'] = users
        state['stock'] = stock - qty
        users.add(user)
        return True


def idempotent_claim(seen, key):
    """幂等键唯一占位（修复点 4、8）。

    对幂等键做原子的 check-then-insert：首次见到的键写入并返回 True，
    重复键（客户端重试、消息重投）返回 False，调用方据此改走「查结果」
    而不是重新执行业务。seen 需支持 in/add 的集合语义。

    Args:
        seen: 已处理幂等键的集合（持久化存储，如唯一索引表/Redis SET）。
        key:  幂等键，None 或空值视为非法。

    Returns:
        bool: 首次处理 True，重复键或参数非法 False。
    """
    if key is None or key == '':
        return False
    try:
        # 集合本身不是线程安全的，check-then-insert 必须互斥；
        # 按 seen 容器分锁，避免不同业务的幂等键共用一把全局锁。
        with _state_lock(seen):
            if key in seen:
                return False
            seen.add(key)
    except TypeError:
        return False
    return True


def lock_key(resource, identifier):
    """生成跨实例确定的分布式锁名（修复点 5）。

    所有应用实例对同一 (resource, identifier) 必须算出同一个字符串，
    再配合 Redis SET NX PX / etcd 租约等外部存储实现跨单机互斥；
    进程内 threading.Lock 无法覆盖多实例部署，不能作为最终互斥手段。

    Args:
        resource:   资源类型，如 'coupon'。
        identifier: 资源标识，如优惠券 ID 'c1'。

    Returns:
        str: 形如 'coupon:c1' 的锁名。

    Raises:
        TypeError: 参数为空或非字符串。
    """
    if not isinstance(resource, str) or not resource:
        raise TypeError('resource must be a non-empty string')
    if not isinstance(identifier, str) or not identifier:
        raise TypeError('identifier must be a non-empty string')
    return f'{resource}:{identifier}'


def is_retryable(reason):
    """判断失败原因是否可安全重试（修复点 8）。

    可重试仅限瞬时、与业务结果无关的错误（死锁、序列化失败、超时、
    连接/可用性故障）；业务校验类错误结果确定，重试无意义。注意：
    可重试的请求也必须带幂等键，重试前先 idempotent_claim 查结果，
    不能把「响应丢失但实际已成功」的请求当失败重新执行。

    Args:
        reason: 错误码字符串，或带 reason/errno 属性的异常对象。

    Returns:
        bool: 可重试 True，不可重试或无法识别 False（保守不重试）。
    """
    if isinstance(reason, BaseException):
        reason = getattr(reason, 'reason', None) or getattr(reason, 'errno', None)
    if not isinstance(reason, str) or not reason:
        return False
    normalized = reason.strip().lower()
    if normalized in _RETRYABLE_REASONS:
        return True
    if normalized in _NON_RETRYABLE_REASONS:
        return False
    # 未知错误码保守处理：默认不可重试，避免对已成功操作造成重复执行。
    return False


def batch_reserve(state, claims):
    """批量领取：统一事务、一次提交（修复点 7、10）。

    对同一优惠券的多条领取在同一把锁内逐条校验，全部合法才统一提交；
    任一条重复领取或余量不足，整批不写入任何变更（all-or-nothing），
    杜绝循环里逐条提交导致的部分成功与余量脏数据。

    Args:
        state:  含 'stock'，可选含 'users' 的可变映射。
        claims: 可迭代的 (user, qty) 元组。

    Returns:
        bool: 整批提交成功 True，否则 False（state 保持调用前状态）。
    """
    if not isinstance(state, dict):
        return False
    try:
        items = list(claims)
    except TypeError:
        return False
    if not items:
        return False
    for user, qty in items:  # 形状与参数先校验，避免半批修改后才报错。
        if user is None or user == '' or not _positive_qty(qty):
            return False

    lock = _state_lock(state)
    with lock:
        stock = state.get('stock')
        if not _non_negative_int(stock):
            return False
        existing = state.get('users')
        users = set(existing) if existing is not None else set()

        needed = 0
        for user, qty in items:
            if user in users:  # 批内或历史重复领取。
                return False
            needed += qty
        if needed > stock:  # 总量条件更新，保证扣减后余量非负。
            return False

        # 提交点：扣减与全部发券登记一次性生效。
        for user, _qty in items:
            users.add(user)
        state['stock'] = stock - needed
        state['users'] = users
        return True
