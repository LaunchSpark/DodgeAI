### Task 7: Lua prelude — tables, math, seeded PRNG

Implemented in Lua rather than Python because these are called thousands of times per frame; crossing the language boundary for `add`/`all` would dominate the frame budget. The PRNG is Lehmer/Park-Miller: not PICO-8's exact generator, but deterministic and seedable, which is what phase 2 needs.

`all()` must tolerate deletion during iteration — the cart does `del(part,_p)` inside `for _p in all(part)`.

**Files:**
- Create: `pico8/prelude.lua`, `tests/test_prelude.py`

**Interfaces:**
- Consumes: nothing
- Produces: a Lua chunk defining `add del all foreach count mid flr ceil abs min max sqrt sgn sin cos rnd srand tostr sub`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prelude.py
from pathlib import Path
import pytest

lupa = pytest.importorskip("lupa")
PRELUDE = Path("pico8/prelude.lua")


@pytest.fixture
def lua():
    runtime = lupa.LuaRuntime(unpack_returned_tuples=True)
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


def test_all_on_empty_and_nil(lua):
    assert lua.execute("local n=0 for v in all({}) do n=n+1 end return n") == 0
    assert lua.execute("local n=0 for v in all(nil) do n=n+1 end return n") == 0


def test_add_returns_value_and_appends(lua):
    assert lua.execute("local t={} add(t,7) add(t,8) return t[1]..','..t[2]") == "7,8"


def test_del_removes_first_match_only(lua):
    assert lua.execute("local t={1,2,1} del(t,1) return #t..':'..t[1]..','..t[2]") == "2:2,1"


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prelude.py -v`
Expected: FAIL with `FileNotFoundError: pico8/prelude.lua`

- [ ] **Step 3: Implement**

```lua
-- pico8/prelude.lua
-- PICO-8 builtins that are pure Lua. Implemented here rather than in Python
-- because they run thousands of times per frame.

function add(t, v)
  if t ~= nil then t[#t + 1] = v end
  return v
end

function del(t, v)
  if t == nil then return end
  for i = 1, #t do
    if t[i] == v then
      table.remove(t, i)
      return v
    end
  end
end

-- Tolerates deletion of the current element during iteration, which the cart
-- relies on (`del(part,_p)` inside `for _p in all(part)`).
function all(t)
  if t == nil or #t == 0 then return function() end end
  local i, prev = 1, nil
  return function()
    if t[i] == prev then i = i + 1 end
    while i <= #t and t[i] == nil do i = i + 1 end
    prev = t[i]
    return t[i]
  end
end

function foreach(t, f)
  for v in all(t) do f(v) end
end

function count(t)
  if t == nil then return 0 end
  return #t
end

flr = math.floor
ceil = math.ceil
abs = math.abs
min = math.min
max = math.max
sqrt = math.sqrt
sub = string.sub

function mid(a, b, c)
  return math.max(math.min(a, b), math.min(math.max(a, b), c))
end

function sgn(x)
  if x < 0 then return -1 end
  return 1
end

-- PICO-8 angles are turns, and its sine is inverted relative to math.sin.
local TAU = 6.283185307179586
function sin(x) return -math.sin(x * TAU) end
function cos(x) return math.cos(x * TAU) end

function tostr(v)
  if v == nil then return "[nil]" end
  return tostring(v)
end

-- Lehmer / Park-Miller. Not PICO-8's generator, but deterministic and seedable.
local seed = 1
function srand(s)
  seed = math.floor(s) % 2147483647
  if seed <= 0 then seed = seed + 2147483646 end
end

function rnd(x)
  seed = (seed * 16807) % 2147483647
  local r = (seed - 1) / 2147483646
  if type(x) == "table" then
    if #x == 0 then return nil end
    return x[math.floor(r * #x) + 1]
  end
  if x == nil then x = 1 end
  return r * x
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prelude.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add pico8/prelude.lua tests/test_prelude.py
git commit -m "feat: Lua prelude with tables, math and seeded PRNG"
```

---

