-- The questions on misata.studio/datasets/factory-sensors-failure-labels, each with the query that answers it.
-- Run in DuckDB from this folder: duckdb < ANSWERS.sql

create view assets as select * from 'assets.csv';
create view failure_events as select * from 'failure_events.csv';
create view sensor_readings as select * from 'sensor_readings.csv';
create view sites as select * from 'sites.csv';
create view work_orders as select * from 'work_orders.csv';

-- 1. How early can a model flag a failure using vibration and temperature alone?
-- Vibration averages 2.57 mm/s over a week out and 4.45 mm/s in the last 24 hours.
-- Running readings only, by hours left before the next failure.
select case when rul_hours <= 24 then '0-24 h'
            when rul_hours <= 72 then '25-72 h'
            when rul_hours <= 168 then '73-168 h'
            else 'over 168 h' end as hours_to_failure,
       count(*) as readings,
       round(avg(vibration_mm_s), 2) as avg_vibration_mm_s,
       round(avg(bearing_temp_c), 1) as avg_bearing_temp_c
from sensor_readings
where operating_mode <> 'stopped' and rul_hours is not null
group by 1
order by min(rul_hours);

-- 2. Does remaining useful life fall faster for some failure modes?
-- seal_leak comes soonest after a repair (1,202 h on average), bearing_wear latest (1,340 h).
select failure_mode,
       count(*) as failures,
       round(avg(hours_since_last_repair), 0) as avg_hours_between_failures,
       round(min(hours_since_last_repair), 0) as shortest,
       round(max(hours_since_last_repair), 0) as longest
from failure_events
where hours_since_last_repair is not null
group by failure_mode
order by avg_hours_between_failures;

-- 3. How much downtime does each failure mode cost?
-- bearing_wear costs the most downtime: 119 hours over 11 failures.
select failure_mode,
       count(*) as failures,
       round(sum(downtime_hours), 1) as downtime_hours,
       round(avg(downtime_hours), 1) as avg_downtime_hours
from failure_events
group by failure_mode
order by downtime_hours desc;

-- 4. Are preventive work orders placed before or after degradation begins?
-- Preventive work orders opened while the machine was healthy (13), degrading (6), critical (5).
-- The machine's state in the hour before each work order was opened (the work order itself stops it).
select w.order_type, r.degradation_stage, count(*) as work_orders
from work_orders w
join sensor_readings r
  on r.asset_id = w.asset_id and r.recorded_at = date_trunc('hour', w.opened_at) - interval '1 hour'
group by w.order_type, r.degradation_stage
order by w.order_type, work_orders desc;
