-- Select all infected palms with health below threshold
SELECT * FROM palms WHERE health_score < 40;

-- Calculate average health by sector
SELECT sector_id, AVG(health_score) 
FROM surveys 
GROUP BY sector_id;
