-- ============================================================
-- 清理 knowledge_documents 表的重复行
-- 背景:上传接口最初无去重,同名文件每次上传都 INSERT 一条新记录。
-- 现在 file.py 改成 upsert 后,需要把历史重复数据清掉。
--
-- 策略:每个 filename 保留"最优的一条":
--   1) 优先保留 status='indexed' 的(已成功建过索引)
--   2) 同状态下保留 created_at 最新的
-- 其它重复行软删(status='deleted'),保留审计痕迹。
--
-- 使用顺序:
--   1) 先跑 STEP 1 看哪些行会被删,人工 review
--   2) 没问题再跑 STEP 2 实际更新
--   3) 可选:STEP 3 同时清理关联的 index_jobs(否则只是软删 document)
-- ============================================================

-- ---------- STEP 1:预览要被软删的行 ----------
WITH ranked AS (
  SELECT
    id,
    filename,
    status,
    version,
    created_at,
    ROW_NUMBER() OVER (
      PARTITION BY filename
      ORDER BY
        CASE WHEN status = 'indexed' THEN 0 ELSE 1 END,
        created_at DESC
    ) AS rn
  FROM knowledge_documents
  WHERE status != 'deleted'
)
SELECT id, filename, status, version, created_at, rn
FROM ranked
WHERE rn > 1
ORDER BY filename, rn;


-- ---------- STEP 2:实际执行软删 ----------
-- 谨慎:确认 STEP 1 的结果后再跑
WITH ranked AS (
  SELECT
    id,
    ROW_NUMBER() OVER (
      PARTITION BY filename
      ORDER BY
        CASE WHEN status = 'indexed' THEN 0 ELSE 1 END,
        created_at DESC
    ) AS rn
  FROM knowledge_documents
  WHERE status != 'deleted'
)
UPDATE knowledge_documents
SET status = 'deleted',
    updated_at = NOW()
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);


-- ---------- STEP 3 (可选):删除软删文档对应的 index_jobs ----------
-- 软删的 document 不会再被处理,对应的 index_jobs 留着没意义,可以一并清掉。
-- 如果想保留任务历史记录就跳过这步。
DELETE FROM index_jobs
WHERE document_id IN (
  SELECT id FROM knowledge_documents WHERE status = 'deleted'
);


-- ---------- 验证 ----------
-- 清理后看一眼,应该每个 filename 只剩一条
SELECT filename, COUNT(*) AS cnt
FROM knowledge_documents
WHERE status != 'deleted'
GROUP BY filename
HAVING COUNT(*) > 1;
-- 上面这条查询应该返回 0 行
