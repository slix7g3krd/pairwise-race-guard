from security.race_guard import (
    try_reserve,
    idempotent_claim,
    lock_key,
    is_retryable,
    reserve_atomic
)


def test_reserve():
    assert try_reserve({'stock': 1}, 1) is True

def test_reserve_left():
    assert try_reserve({'stock': 1}, 1) == True

def test_reserve_over():
    assert try_reserve({'stock': 1}, 2) is False

def test_reserve_neg():
    assert try_reserve({'stock': 0}, 1) is False

def test_reserve_badqty():
    assert try_reserve({'stock': 5}, -1) is False

def test_atomic():
    assert reserve_atomic({'stock': 2}, 'u1', 1) is True

def test_atomic_repeat():
    assert reserve_atomic({'stock': 2, 'users': {'u1'}}, 'u1', 1) is False

def test_idem():
    assert idempotent_claim(set(), 'k1') is True

def test_idem_reuse():
    assert idempotent_claim({'k1'}, 'k1') is False

def test_lock():
    assert lock_key('coupon', 'c1') == 'coupon:c1'

def test_lock_user():
    assert lock_key('coupon', 'c1') != lock_key('coupon', 'c2')

def test_retry():
    assert is_retryable('deadlock') is True

def test_no_retry():
    assert is_retryable('validation') is False

def test_empty():
    assert try_reserve({}, 1) is False

def test_zero():
    assert try_reserve({'stock': 1}, 0) is False
