SELECT 
    s.subject_name AS 学科,
    i.institution_name AS 学校,
    m.rank AS 排名,
    m.cites_per_paper AS 平均被引,
    m.top_papers AS 高被引论文数
FROM metrics m
JOIN subjects s ON m.subject_id = s.subject_id
JOIN institutions i ON m.institution_id = i.institution_id
WHERE i.country = 'CHINA MAINLAND'
ORDER BY s.subject_name, m.rank;
