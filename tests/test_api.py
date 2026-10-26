from tests.conftest import KEY


def upload(client, pdf, name="handbook.pdf", headers=KEY):
    return client.post("/documents", files={"file": (name, pdf, "application/pdf")}, headers=headers)


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client, handbook_pdf):
    assert client.post("/ask", json={"question": "refund?"}).status_code == 401
    assert upload(client, handbook_pdf, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents").status_code == 401


def test_upload_ask_with_citation(client, handbook_pdf):
    r = upload(client, handbook_pdf)
    assert r.status_code == 201 and r.json()["status"] == "indexed" and r.json()["pages"] == 2
    res = client.post("/ask", json={"question": "What is your refund policy?"}, headers=KEY).json()
    assert res["grounded"] is True
    assert "30 days" in res["answer"]
    assert res["sources"][0]["filename"] == "handbook.pdf" and res["sources"][0]["page"] == 1


def test_page_citation_for_second_page(client, handbook_pdf):
    upload(client, handbook_pdf)
    res = client.post("/ask", json={"question": "How long is the warranty?"}, headers=KEY).json()
    assert res["sources"][0]["page"] == 2


def test_out_of_scope_question_is_refused(client, handbook_pdf):
    upload(client, handbook_pdf)
    res = client.post("/ask", json={"question": "Who is the president of Brazil?"}, headers=KEY).json()
    assert res["grounded"] is False and res["sources"] == []


def test_ask_with_empty_knowledge_base(client):
    res = client.post("/ask", json={"question": "What is your refund policy?"}, headers=KEY).json()
    assert res["grounded"] is False


def test_duplicate_upload_is_idempotent(client, handbook_pdf):
    upload(client, handbook_pdf)
    assert upload(client, handbook_pdf).json()["status"] == "already_indexed"
    assert len(client.get("/documents", headers=KEY).json()["documents"]) == 1


def test_delete_document(client, handbook_pdf):
    doc_id = upload(client, handbook_pdf).json()["doc_id"]
    assert client.delete(f"/documents/{doc_id}", headers=KEY).status_code == 200
    assert client.delete(f"/documents/{doc_id}", headers=KEY).status_code == 404


def test_rejects_non_pdf_extension(client):
    assert upload(client, b"hello", name="notes.txt").status_code == 415


def test_rejects_fake_pdf(client):
    assert upload(client, b"not really a pdf", name="fake.pdf").status_code == 415


def test_rejects_corrupt_pdf(client):
    assert upload(client, b"%PDF-1.4 garbage garbage", name="bad.pdf").status_code == 422


def test_rejects_empty_file(client):
    assert upload(client, b"", name="empty.pdf").status_code == 400


def test_rejects_oversized_file(client, settings, handbook_pdf):
    settings.max_upload_mb = 0
    assert upload(client, handbook_pdf).status_code == 413


def test_question_validation(client):
    assert client.post("/ask", json={"question": "a"}, headers=KEY).status_code == 422
