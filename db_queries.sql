-- 1. Spread + tick volatility by UTC hour
SELECT CAST(SUBSTR(utc_time,1,2) AS INT) AS hh, COUNT(*) AS n,
       AVG(market_neutral_spread_pip) AS avg_spr,
       MIN(market_neutral_spread_pip) AS min_spr,
       MAX(market_neutral_spread_pip) AS max_spr
FROM fx_algo_data WHERE type='RATE' GROUP BY hh ORDER BY hh;

-- 2. Mid move per tick by hour (in pips)
SELECT hh, AVG(ABS(d))*10000 AS avg_abs_move_pips, COUNT(*) n FROM (
  SELECT CAST(SUBSTR(utc_time,1,2) AS INT) AS hh,
         market_neutral_mid - LAG(market_neutral_mid) OVER (ORDER BY ts) AS d
  FROM fx_algo_data WHERE type='RATE') t
WHERE d IS NOT NULL GROUP BY hh ORDER BY hh;

-- 3. Request arrival rate + size by hour
SELECT CAST(SUBSTR(utc_time,1,2) AS INT) AS hh, COUNT(*) n,
       AVG(size_m) avg_sz, MIN(size_m) min_sz, MAX(size_m) max_sz
FROM fx_algo_data WHERE type='REQUEST' GROUP BY hh ORDER BY hh;

-- 4. THE IMPORTANT ONE: limit price aggressiveness vs mid
SELECT side, ROUND(agg_pips,1) AS pips_from_mid, COUNT(*) n FROM (
  SELECT r.side,
         (r.limit_price - (SELECT market_neutral_mid FROM fx_algo_data m
                           WHERE m.type='RATE' AND m.ts <= r.ts
                           ORDER BY m.ts DESC LIMIT 1)) * 10000 AS agg_pips
  FROM fx_algo_data r WHERE r.type='REQUEST') t
GROUP BY side, pips_from_mid ORDER BY side, pips_from_mid;