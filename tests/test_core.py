from compute_infinity import hello_world


def test_hello_world_returns_expected_string() -> None:
    assert hello_world() == "hello, world"
