-- SQL Script to delete malicious search logs (SQL Injection, XSS, Spam)
-- Execute in MySQL Database

DELETE FROM search_log_products 
WHERE search_log_id IN (
    SELECT id FROM search_logs 
    WHERE LOWER(query_text) LIKE '%or 1=1%'
       OR LOWER(query_text) LIKE '%and 1=1%'
       OR LOWER(query_text) LIKE '%1=1--%'
       OR LOWER(query_text) LIKE '%-- -%'
       OR LOWER(query_text) LIKE '%union select%'
       OR LOWER(query_text) LIKE '%<script%'
       OR LOWER(query_text) LIKE '%drop table%'
       OR LOWER(query_text) LIKE '%sleep(%'
       OR LOWER(keyword) LIKE '%or 1=1%'
       OR LOWER(keyword) LIKE '%and 1=1%'
);

DELETE FROM search_logs 
WHERE LOWER(query_text) LIKE '%or 1=1%'
   OR LOWER(query_text) LIKE '%and 1=1%'
   OR LOWER(query_text) LIKE '%1=1--%'
   OR LOWER(query_text) LIKE '%-- -%'
   OR LOWER(query_text) LIKE '%union select%'
   OR LOWER(query_text) LIKE '%<script%'
   OR LOWER(query_text) LIKE '%drop table%'
   OR LOWER(query_text) LIKE '%sleep(%'
   OR LOWER(keyword) LIKE '%or 1=1%'
   OR LOWER(keyword) LIKE '%and 1=1%'
   OR query_text REGEXP '^[[:punct:]]+$'
   OR query_text IS NULL 
   OR TRIM(query_text) = '';
