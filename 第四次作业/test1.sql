SELECT 
    s.subject_name AS 学科,
    m.rank AS 排名,
    m.wos_documents AS 论文数,
    m.cites AS 被引次数,
    m.cites_per_paper AS 平均被引,
    m.top_papers AS 高被引论文数
FROM metrics m
JOIN subjects s ON m.subject_id = s.subject_id
JOIN institutions i ON m.institution_id = i.institution_id
WHERE i.institution_name LIKE '%EAST CHINA NORMAL UNIVERSITY%'
ORDER BY m.rank;
