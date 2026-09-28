"""Bind one independently read shared source to its thirteen declared runners."""

from dataclasses import replace

from amp_challenge.generators.search import durable_dispatch_journal_records as j
from amp_challenge.generators.search.common_initial_source_records import SourcePlan
from amp_challenge.generators.search.common_initial_source_verify import (
    verify_common_initial_source,
)


def load_common_initial_handoffs(
    bindings,
    root,
    *,
    expected_plan,
    expected_plan_sha256,
    expected_head_sha256,
    repository,
    expected_sources,
    authenticator,
):
    """Return (binding, source receipt, copy receipt) for each declared destination.

    Provisioners call this before constructing their feature bridge and runner
    inputs. Existing runners import the returned receipts into their own journals.
    This function makes no submissions, imports or timing/resource attestations.
    It changes only the two receipt hashes. Destination, schedules and authority
    must match the prospective source; the destination timing remains unchanged.
    """
    j.require(type(expected_plan) is SourcePlan, "exact source plan required")
    expected_plan.__post_init__()
    j.require(
        type(bindings) is tuple
        and len(bindings) == 13
        and all(type(binding) is j.JournalBinding for binding in bindings),
        "thirteen exact destination bindings required",
    )
    reserves = tuple(
        expected_plan.reserve_requests[2 * index : 2 * index + 2] for index in range(28)
    )
    for binding, (arm_id, run_id) in zip(bindings, expected_plan.destinations, strict=True):
        binding.__post_init__()
        j.require(
            (binding.arm_id, binding.run_id) == (arm_id, run_id)
            and binding.seed == expected_plan.seed
            and binding.objective_context_sha256 == expected_plan.objective_context_sha256
            and binding.oracle_bundle_sha256 == expected_plan.oracle_bundle_sha256
            and binding.initial_source_run_id == expected_plan.source_run_id
            and binding.initial_requests == expected_plan.initial_requests
            and binding.reserves == reserves
            and binding.authenticator_sha256 == authenticator.source_sha256,
            "destination identity, schedule or source authority differs",
        )
    readback = verify_common_initial_source(
        root,
        expected_plan=expected_plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_head_sha256=expected_head_sha256,
        repository=repository,
        expected_sources=expected_sources,
        authenticator=authenticator,
    )
    source = bytes.fromhex(readback["source_receipt_hex"])
    copies = tuple(bytes.fromhex(value) for value in readback["copy_receipt_hex"])
    return tuple(
        (
            replace(
                binding,
                initial_source_receipt_sha256=j.digest(source),
                initial_copy_receipt_sha256=j.digest(copy),
            ),
            source,
            copy,
        )
        for binding, copy in zip(bindings, copies, strict=True)
    )


def load_common_initial_handoff(
    binding,
    root,
    *,
    expected_plan,
    expected_plan_sha256,
    expected_head_sha256,
    repository,
    expected_sources,
    authenticator,
):
    """Read the full source and return one destination's binding/source/copy.

    The caller owns only this destination's original clock. No peer bindings
    are constructed; only this binding's two receipt hashes change.
    """
    j.require(type(expected_plan) is SourcePlan, "exact source plan required")
    expected_plan.__post_init__()
    j.require(type(binding) is j.JournalBinding, "exact destination binding required")
    binding.__post_init__()
    destination = (binding.arm_id, binding.run_id)
    reserves = tuple(
        expected_plan.reserve_requests[2 * index : 2 * index + 2] for index in range(28)
    )
    j.require(
        destination in expected_plan.destinations
        and binding.seed == expected_plan.seed
        and binding.objective_context_sha256 == expected_plan.objective_context_sha256
        and binding.oracle_bundle_sha256 == expected_plan.oracle_bundle_sha256
        and binding.initial_source_run_id == expected_plan.source_run_id
        and binding.initial_requests == expected_plan.initial_requests
        and binding.reserves == reserves
        and binding.authenticator_sha256 == authenticator.source_sha256,
        "destination identity, schedule or source authority differs",
    )
    readback = verify_common_initial_source(
        root,
        expected_plan=expected_plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_head_sha256=expected_head_sha256,
        repository=repository,
        expected_sources=expected_sources,
        authenticator=authenticator,
    )
    source = bytes.fromhex(readback["source_receipt_hex"])
    copy = bytes.fromhex(
        readback["copy_receipt_hex"][expected_plan.destinations.index(destination)]
    )
    return (
        replace(
            binding,
            initial_source_receipt_sha256=j.digest(source),
            initial_copy_receipt_sha256=j.digest(copy),
        ),
        source,
        copy,
    )
