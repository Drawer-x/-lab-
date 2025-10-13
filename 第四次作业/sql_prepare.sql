CREATE DATABASE IF NOT EXISTS university_data
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE university_data;

CREATE TABLE IF NOT EXISTS disciplines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject VARCHAR(255),
    institution VARCHAR(255),
    `rank` INT,
    wos_documents INT,
    cites INT,
    cites_per_paper FLOAT,
    top_papers INT
);
