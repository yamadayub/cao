-- Remove foreign key constraint on user_id since Clerk users may not exist in profiles table
-- The user_id is stored for reference but doesn't need referential integrity

ALTER TABLE sns_shares DROP CONSTRAINT IF EXISTS sns_shares_user_id_fkey;
