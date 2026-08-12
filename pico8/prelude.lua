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
