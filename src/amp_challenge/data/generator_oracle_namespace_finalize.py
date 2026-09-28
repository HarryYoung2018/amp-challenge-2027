"""Finalize a namespace audit only after authoritative Slurm completion exists."""

from __future__ import annotations

import argparse
from pathlib import Path

from amp_challenge.data.generator_oracle_namespace_verify import (
    CLAIMS,
    GIT_PATTERN,
    JOB_PATTERN,
    NODE_PATTERN,
    SHA_PATTERN,
    _atomic_publish_receipt,
    _authenticate_producer_commit,
    _authenticate_repository,
    _canonical,
    _fail,
    _read_committed_receipt,
    _sacct_completed_job,
    _verify_runtime_attestation,
)


def finalize(
    *,
    repository_root: Path,
    expected_git_commit: str,
    producer_job_id: str,
    audit_job_id: str,
    finalizer_job_id: str,
    finalizer_node: str,
    candidate_receipt: Path,
    audit_environment: Path,
    audit_runtime_attestation: Path,
    finalizer_environment: Path,
    finalizer_runtime_attestation: Path,
    finalizer_runtime_environment_sha256: str,
    finalizer_runtime_attestation_marker_sha256: str,
    output_receipt: Path,
) -> None:
    _fail(bool(GIT_PATTERN.fullmatch(expected_git_commit)), "finalizer commit is malformed")
    _fail(
        all(
            bool(JOB_PATTERN.fullmatch(value))
            for value in (producer_job_id, audit_job_id, finalizer_job_id)
        ),
        "finalizer job identity is malformed",
    )
    _fail(bool(NODE_PATTERN.fullmatch(finalizer_node)), "finalizer node is malformed")
    _fail(
        bool(SHA_PATTERN.fullmatch(finalizer_runtime_environment_sha256))
        and bool(SHA_PATTERN.fullmatch(finalizer_runtime_attestation_marker_sha256)),
        "finalizer runtime binding is malformed",
    )
    source_inventory = _authenticate_repository(repository_root, expected_git_commit)
    candidate, candidate_image = _read_committed_receipt(candidate_receipt)
    _fail(
        isinstance(candidate, dict)
        and candidate.get("schema_version") == 2
        and candidate.get("artifact")
        == "generator_oracle_namespace_split_independent_verification_candidate_v2"
        and candidate.get("status")
        == "verification_passed_candidate_pending_scheduler_finalization"
        and candidate.get("claims") == CLAIMS,
        "audit candidate schema/status mismatch",
    )
    identity = candidate.get("identity")
    _fail(isinstance(identity, dict), "audit candidate identity is malformed")
    _fail(
        identity.get("audit_job_id") == audit_job_id
        and identity.get("producer_job_id") == producer_job_id
        and identity.get("audit_git_commit") == expected_git_commit
        and identity.get("producer_git_commit") == expected_git_commit
        and identity.get("audit_source_inventory") == source_inventory,
        "audit candidate identity/source mismatch",
    )
    producer_inventory = _authenticate_producer_commit(
        repository_root,
        identity["producer_git_commit"],
        expected_git_commit,
    )
    _fail(
        identity.get("producer_source_inventory") == producer_inventory,
        "audit candidate producer source inventory mismatch",
    )
    _fail(
        isinstance(candidate.get("checks"), dict)
        and candidate["checks"]
        and all(value is True for value in candidate["checks"].values()),
        "audit candidate did not pass every content check",
    )
    state = candidate.get("candidate_state")
    markers = candidate.get("validity_markers")
    _fail(
        state
        == {
            "content_verification_passed": True,
            "authoritative_audit_sacct_pending": True,
            "formal_commit_point": "single_link_candidate_receipt",
            "final_receipt_required": True,
        }
        and isinstance(markers, dict)
        and markers.get("content_verification_passed") is True
        and markers.get("authoritative_scheduler_finalization_passed") is False
        and markers.get("independent_verification_passed") is False,
        "audit candidate state is ambiguous",
    )
    producer_scheduler = _sacct_completed_job(
        producer_job_id,
        expected_job_name="amp-ns-split-v2",
        expected_alloc_cpus=4,
        expected_nodes=2,
        expected_total_memory_mib=16_384,
    )
    audit_scheduler = _sacct_completed_job(
        audit_job_id,
        expected_job_name="amp-ns-split-audit-v2",
        expected_alloc_cpus=4,
        expected_nodes=1,
        expected_total_memory_mib=16_384,
    )
    _fail(
        candidate.get("producer_scheduler") == producer_scheduler,
        "producer accounting changed between audit and finalization",
    )
    excluded = identity.get("scheduler_excluded_producer_nodes")
    _fail(
        isinstance(excluded, list)
        and set(excluded) == set(producer_scheduler["nodes"])
        and audit_scheduler["nodes"] == [identity.get("audit_node")]
        and not set(audit_scheduler["nodes"]) & set(producer_scheduler["nodes"]),
        "filesystem node records, exclusions, and authoritative accounting disagree",
    )
    audit_runtime = _verify_runtime_attestation(
        environment=audit_environment,
        attestation=audit_runtime_attestation,
        expected_inventory_sha256=identity["audit_runtime_environment_sha256"],
        expected_marker_sha256=identity["audit_runtime_attestation_marker_sha256"],
        expected_commit=expected_git_commit,
        expected_job_id=audit_job_id,
        source_inventory=source_inventory,
        role="audit",
    )
    finalizer_runtime = _verify_runtime_attestation(
        environment=finalizer_environment,
        attestation=finalizer_runtime_attestation,
        expected_inventory_sha256=finalizer_runtime_environment_sha256,
        expected_marker_sha256=finalizer_runtime_attestation_marker_sha256,
        expected_commit=expected_git_commit,
        expected_job_id=finalizer_job_id,
        source_inventory=source_inventory,
        role="finalizer",
    )
    final_receipt = {
        "schema_version": 2,
        "artifact": "generator_oracle_namespace_split_independent_verification_v2",
        "status": "passed_non_authorizing_data_preparation_only",
        "claims": CLAIMS,
        "candidate": {
            "sha256": candidate_image.sha256,
            "dev": candidate_image.fingerprint[0],
            "ino": candidate_image.fingerprint[1],
            "bytes": candidate_image.size,
        },
        "identity": {
            **identity,
            "finalizer_job_id": finalizer_job_id,
            "finalizer_node": finalizer_node,
            "finalizer_git_commit": expected_git_commit,
            "finalizer_runtime_environment_sha256": finalizer_runtime_environment_sha256,
            "finalizer_runtime_attestation_marker_sha256": (
                finalizer_runtime_attestation_marker_sha256
            ),
        },
        "producer_scheduler": producer_scheduler,
        "audit_scheduler": audit_scheduler,
        "audit_runtime": audit_runtime,
        "finalizer_runtime": finalizer_runtime,
        "checks": candidate["checks"],
        "census": candidate["census"],
        "artifact_records": candidate["artifact_records"],
        "overlap": candidate["overlap"],
        "fold_triples": candidate["fold_triples"],
        "provenance_invariant_output_names": candidate["provenance_invariant_output_names"],
        "twins": candidate["twins"],
        "publication_probe": candidate["publication_probe"],
        "receipt_state": {
            "candidate_content_verified": True,
            "authoritative_producer_sacct_verified": True,
            "authoritative_audit_sacct_verified": True,
            "formal_commit_point": "single_link_final_receipt",
        },
        "validity_markers": {
            "staging_authenticated": True,
            "namespace_reconstruction_passed": True,
            "producer_twins_passed": True,
            "content_verification_passed": True,
            "authoritative_scheduler_finalization_passed": True,
            "independent_verification_passed": True,
            "downstream_execution_authorized": False,
            "scientific_evidence_accepted": False,
            "production_input_eligible": False,
        },
    }
    payload = _canonical(final_receipt)
    _fail(b"/lustre/" not in payload and b"/home/" not in payload, "final receipt leaks a path")
    # Finalization can follow a queued audit. Recheck the small candidate and
    # repository after accounting/runtime reads so a cooperative workflow cannot
    # accidentally finalize stale evidence after replacing either input.
    rebound_candidate, rebound_image = _read_committed_receipt(candidate_receipt)
    _fail(
        rebound_candidate == candidate and rebound_image.fingerprint == candidate_image.fingerprint,
        "audit candidate changed during finalization",
    )
    _fail(
        _authenticate_repository(repository_root, expected_git_commit) == source_inventory,
        "finalizer source inventory changed before publication",
    )
    # This must remain the final operation: publication's single-link transition
    # is the receipt commit point and no success/failure ambiguity follows it.
    _atomic_publish_receipt(output_receipt, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--producer-job-id", required=True)
    parser.add_argument("--audit-job-id", required=True)
    parser.add_argument("--finalizer-job-id", required=True)
    parser.add_argument("--finalizer-node", required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--audit-environment", type=Path, required=True)
    parser.add_argument("--audit-runtime-attestation", type=Path, required=True)
    parser.add_argument("--finalizer-environment", type=Path, required=True)
    parser.add_argument("--finalizer-runtime-attestation", type=Path, required=True)
    parser.add_argument("--finalizer-runtime-environment-sha256", required=True)
    parser.add_argument("--finalizer-runtime-attestation-marker-sha256", required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    finalize(
        repository_root=args.repository_root,
        expected_git_commit=args.expected_git_commit,
        producer_job_id=args.producer_job_id,
        audit_job_id=args.audit_job_id,
        finalizer_job_id=args.finalizer_job_id,
        finalizer_node=args.finalizer_node,
        candidate_receipt=args.candidate_receipt,
        audit_environment=args.audit_environment,
        audit_runtime_attestation=args.audit_runtime_attestation,
        finalizer_environment=args.finalizer_environment,
        finalizer_runtime_attestation=args.finalizer_runtime_attestation,
        finalizer_runtime_environment_sha256=args.finalizer_runtime_environment_sha256,
        finalizer_runtime_attestation_marker_sha256=(
            args.finalizer_runtime_attestation_marker_sha256
        ),
        output_receipt=args.output_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
