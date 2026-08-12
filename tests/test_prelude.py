from pathlib import Path

import pytest

luajit = pytest.importorskip("lupa.luajit21")
PRELUDE = Path("pico8/prelude.lua")


@pytest.fixture
def lua():
    runtime = luajit.LuaRuntime(unpack_returned_tuples=True)
    runtime.execute(PRELUDE.read_text(encoding="utf-8"))
    return runtime


def test_all_survives_deletion_during_iteration(lua):
    result = lua.execute("""
        local t = {1,2,3,4,5}
        local seen = {}
        for v in all(t) do
            seen[#seen+1] = v
            if v % 2 == 0 then del(t, v) end
        end
        return table.concat(seen, ",")
    """)
    assert result == "1,2,3,4,5"


def test_all_survives_deletion_of_first_element(lua):
    result = lua.execute("""
        local t = {1,2,3}
        local seen = {}
        for v in all(t) do
            seen[#seen+1] = v
            if v == 1 then del(t, v) end
        end
        return table.concat(seen, ",")
    """)
    assert result == "1,2,3"


def test_all_survives_deletion_of_last_element(lua):
    result = lua.execute("""
        local t = {1,2,3}
        local seen = {}
        for v in all(t) do
            seen[#seen+1] = v
            if v == 3 then del(t, v) end
        end
        return table.concat(seen, ",")
    """)
    assert result == "1,2,3"


def test_all_on_empty_and_nil(lua):
    assert lua.execute("local n=0 for v in all({}) do n=n+1 end return n") == 0
    assert lua.execute("local n=0 for v in all(nil) do n=n+1 end return n") == 0


def test_add_returns_value_and_appends(lua):
    assert lua.execute("local t={} add(t,7) add(t,8) return t[1]..','..t[2]") == "7,8"


def test_del_removes_first_match_only(lua):
    assert lua.execute("local t={1,2,1} del(t,1) return #t..':'..t[1]..','..t[2]") == "2:2,1"


def test_foreach_calls_function_on_each_element(lua):
    assert lua.execute("""
        local t = {1,2,3}
        local sum = 0
        foreach(t, function(v) sum = sum + v end)
        return sum
    """) == 6


def test_count_returns_length(lua):
    assert lua.execute("return count({1,2,3})") == 3
    assert lua.execute("return count(nil)") == 0


def test_mid_clamps_like_pico8(lua):
    assert lua.execute("return mid(2,200,125)") == 125
    assert lua.execute("return mid(2,-5,125)") == 2
    assert lua.execute("return mid(2,50,125)") == 50


def test_sgn_of_zero_is_one(lua):
    assert lua.execute("return sgn(0)") == 1
    assert lua.execute("return sgn(-3)") == -1


def test_sin_is_inverted_and_turn_based(lua):
    assert lua.execute("return sin(0)") == pytest.approx(0, abs=1e-9)
    assert lua.execute("return sin(0.25)") == pytest.approx(-1, abs=1e-9)
    assert lua.execute("return cos(0)") == pytest.approx(1, abs=1e-9)
    assert lua.execute("return cos(0.25)") == pytest.approx(0, abs=1e-9)


def test_rnd_is_deterministic_for_a_seed(lua):
    first = lua.execute("srand(7) local s='' for i=1,5 do s=s..flr(rnd(100))..',' end return s")
    again = lua.execute("srand(7) local s='' for i=1,5 do s=s..flr(rnd(100))..',' end return s")
    assert first == again


def test_rnd_differs_between_seeds(lua):
    a = lua.execute("srand(1) return rnd(1000)")
    b = lua.execute("srand(2) return rnd(1000)")
    assert a != b


def test_rnd_range_and_table_form(lua):
    assert lua.execute("srand(1) local m=0 for i=1,200 do m=math.max(m,rnd(4)) end return m") < 4
    assert lua.execute("srand(1) return rnd({9,9,9})") == 9
