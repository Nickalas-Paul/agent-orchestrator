-- Prevent modification or deletion of audit log entries.
-- Application-level immutability (AuditLogger only INSERTs) is now
-- enforced at the database level. This closes audit finding P3.

CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Modification of audit log entries is not permitted. '
                    'Table agent_audit_logs is append-only.';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Drop existing triggers if re-running
DROP TRIGGER IF EXISTS no_audit_update ON agent_audit_logs;
DROP TRIGGER IF EXISTS no_audit_delete ON agent_audit_logs;

CREATE TRIGGER no_audit_update
    BEFORE UPDATE ON agent_audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_modification();

CREATE TRIGGER no_audit_delete
    BEFORE DELETE ON agent_audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_modification();
