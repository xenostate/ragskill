from scripts.migrate import checksum, migration_files, migration_sql


def test_migrations_are_ordered_and_include_operational_state():
    files = migration_files()

    assert [path.name for path in files] == sorted(path.name for path in files)
    assert "create table if not exists indexing_jobs" in migration_sql(files[-1]).lower()


def test_baseline_include_is_expanded_and_checksummed():
    baseline = migration_files()[0]

    assert "create table if not exists sites" in migration_sql(baseline).lower()
    assert len(checksum(baseline)) == 64
