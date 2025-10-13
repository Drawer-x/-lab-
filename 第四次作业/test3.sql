SELECT 
    s.subject_name AS 学科,
    i.region AS 区域,
    AVG(m.cites_per_paper) AS 平均被引,
    AVG(m.top_papers) AS 平均高被引论文数,
    SUM(m.wos_documents) AS 总论文数
FROM metrics m
JOIN subjects s ON m.subject_id = s.subject_id
JOIN institutions i ON m.institution_id = i.institution_id
GROUP BY s.subject_name, i.region
ORDER BY s.subject_name, i.region;
