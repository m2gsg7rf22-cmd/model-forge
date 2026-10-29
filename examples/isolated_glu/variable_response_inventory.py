"""Pure scalar read/storage bounds; no files, arrays, RNG or experiment."""

from __future__ import annotations


def inventory() -> dict[str, int]:
    pairs = 6 * 2 * 17
    archives = 2 * pairs
    encoded = 544768
    transport = 2 * encoded + 4096
    metadata = 64 * 128 * 1024
    return dict(
        pairs=pairs,
        archives=archives,
        vector_rows=pairs * 2112,
        raw_payload_bytes=archives * 2112 * 32 * 8,
        encoded_pass_upper_bound=archives * encoded,
        complete_archive_reads=3 * archives,
        complete_array_decodes=2 * archives,
        encoded_read_bytes_upper_bound=3 * archives * encoded,
        transport_upper_bound=transport,
        metadata_reserve=metadata,
        new_artifacts_upper_bound=transport + metadata,
        combined_artifacts_upper_bound=534980568 + 135203053 + transport + metadata,
        descriptor_record_upper_bound=512,
        descriptor_seal_upper_bound=pairs * 513 + 2048,
        delivery_record_upper_bound=512,
        delivery_upper_bound=pairs * 512,
        report_cap=128 * 1024,
        transport_manifest_cap=4096,
    )
