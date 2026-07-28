from __future__ import annotations

from rag_app.rag import extract_document_ids, route_by_rules, router_signals


def test_router_detects_document_ids():
    assert extract_document_ids("document DOC-51493") == ["51493"]
    assert extract_document_ids("tài liệu policy-2026") == ["policy-2026"]


def test_router_signals_detect_generic_identifiers():
    signals = router_signals("API order-v2 trả về lỗi 400")

    assert signals["identifier_count"] == 1
    assert signals["error_code_count"] == 1


def test_router_routes_exact_technical_queries_to_bm25():
    mode, reason = route_by_rules('API order-v2 trả về lỗi "INVALID_STATE"')

    assert mode == "BM25"
    assert "exact technical" in reason


def test_router_routes_exact_document_id_to_bm25():
    mode, reason = route_by_rules("document DOC-51493")

    assert mode == "BM25"
    assert "document id" in reason


def test_router_routes_exact_identifier_to_bm25():
    mode, reason = route_by_rules("Xem quy định POLICY-2026-07")

    assert mode == "BM25"
    assert "exact technical" in reason


def test_router_routes_keyword_plus_semantic_to_hybrid():
    mode, reason = route_by_rules(
        "Tìm các tài liệu liên quan đến API nhưng người dùng không nhận được phản hồi"
    )

    assert mode == "Hybrid RRF"
    assert "keyword plus semantic" in reason


def test_router_routes_paraphrase_to_semantic():
    mode, reason = route_by_rules(
        "Nguoi dung sua du lieu nhung he thong khong ghi nhan thay doi, co bai nao giong vay khong?"
    )

    assert mode == "Semantic"
    assert "paraphrased" in reason


def test_router_routes_diverse_requests_to_mmr():
    mode, reason = route_by_rules("Liệt kê các nhóm hướng dẫn khác nhau thường gặp")

    assert mode == "MMR"
    assert "diverse" in reason


def test_router_routes_multi_part_questions_to_decomposition():
    mode, reason = route_by_rules(
        "So sánh hai chính sách: mục đích, đối tượng áp dụng và ngoại lệ"
    )

    assert mode == "Decomposition"
    assert "multi-part" in reason


def test_router_routes_conceptual_queries_to_hyde():
    mode, reason = route_by_rules(
        "Khi quy trình hoạt động khác kỳ vọng thì nên tìm dạng tài liệu nào?"
    )

    assert mode == "HyDE"
    assert "conceptual" in reason
