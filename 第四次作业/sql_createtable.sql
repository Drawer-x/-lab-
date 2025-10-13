CREATE DATABASE IF NOT EXISTS university_analysis
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
USE university_analysis;

CREATE TABLE subjects (
    subject_id INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(255) UNIQUE
);

CREATE TABLE institutions (
    institution_id INT AUTO_INCREMENT PRIMARY KEY,
    institution_name VARCHAR(255),
    country VARCHAR(255),
    region VARCHAR(255)
);

CREATE TABLE metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    subject_id INT,
    institution_id INT,
    `rank` INT,
    wos_documents INT,
    cites INT,
    cites_per_paper FLOAT,
    top_papers INT,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    FOREIGN KEY (institution_id) REFERENCES institutions(institution_id)
);
