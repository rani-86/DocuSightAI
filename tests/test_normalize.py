from docusight import normalize as n


def test_amount_indian_grouping_and_currency():
    assert n.amount("Rs. 1,23,456.5") == "123456.50"
    assert n.amount("₹ 2,268.00") == "2268.00"
    assert n.amount("no number") is None


def test_dates_in_many_formats():
    assert n.iso_date("14/03/2025") == "2025-03-14"
    assert n.iso_date("05 Jan 2026") == "2026-01-05"
    assert n.iso_date("2024-12-01") == "2024-12-01"
    assert n.period("01 Feb 2023 to 28 Feb 2023") == "2023-02-01..2023-02-28"
    assert n.month("Salary Month: Mar-2024") == "2024-03"


def test_position_aware_id_correction():
    assert n.pan("ABCPE1Z34F") == "ABCPE1234F"  # Z->2 in a digit slot
    assert n.pan("A8CPE1234F") == "ABCPE1234F"  # 8->B in a letter slot
    assert n.ifsc("HDFCO001234") == "HDFC0001234"
    assert n.gstin("27ABCPE1234F1Z5") == "27ABCPE1234F1Z5"
