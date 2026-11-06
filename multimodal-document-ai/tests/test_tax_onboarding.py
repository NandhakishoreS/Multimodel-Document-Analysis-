"""
Unit Tests for Phase 2 Taxpayer Onboarding Service.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    FilingStatus,
    ResidentialStatus,
    TaxRegime,
)
from tax_module.db import TaxDatabase
from tax_module.onboarding import TaxpayerOnboardingService


@pytest.fixture
def onboarding_service(tmp_path: Path) -> TaxpayerOnboardingService:
    db = TaxDatabase(db_path=tmp_path / "onboarding_test.db")
    return TaxpayerOnboardingService(db=db)


def test_onboard_valid_individual_taxpayer(onboarding_service: TaxpayerOnboardingService):
    valid_data = {
        "pan": "ABCPE1234F",
        "first_name": "Suresh",
        "middle_name": "K.",
        "last_name": "Rao",
        "dob": "1985-04-12",
        "gender": "M",
        "residential_status": ResidentialStatus.RESIDENT.value,
        "email": "suresh.rao@example.com",
        "mobile": "9845012345",
        "address_flat_door_block": "Plot 12",
        "address_premises_building": "Sai Krupa",
        "address_city": "Hyderabad",
        "address_state": "Telangana",
        "address_pincode": "500001",
    }
    profile, errors = onboarding_service.onboard_taxpayer(valid_data)
    assert errors == []
    assert profile is not None
    assert profile.pan == "ABCPE1234F"
    assert profile.full_name == "Suresh K. Rao"
    assert profile.masked_pan == "ABXXXXXX4F"

    # Verify audit event
    audit = onboarding_service.db.get_audit_trail(taxpayer_id=profile.taxpayer_id)
    assert len(audit) >= 1
    assert audit[0].action == "TAXPAYER_ONBOARDED"


def test_onboard_invalid_pan_scenarios(onboarding_service: TaxpayerOnboardingService):
    # Non-individual PAN (4th letter 'C' = Company)
    company_data = {
        "pan": "ABCDE1234C",
        "first_name": "Apex",
        "last_name": "Corp",
        "dob": "2010-01-01",
    }
    profile, errors = onboarding_service.onboard_taxpayer(company_data)
    assert profile is None
    assert any("Individual" in err for err in errors)

    # Malformed PAN
    bad_pan_data = {
        "pan": "INVALID123",
        "first_name": "Test",
        "last_name": "User",
        "dob": "1990-01-01",
    }
    profile, errors = onboarding_service.onboard_taxpayer(bad_pan_data)
    assert profile is None
    assert any("Invalid PAN format" in err for err in errors)


def test_onboard_invalid_dates_and_contacts(onboarding_service: TaxpayerOnboardingService):
    bad_data = {
        "pan": "ABCPE1234F",
        "first_name": "Priya",
        "last_name": "Nair",
        "dob": "2099-01-01",  # Future date
        "mobile": "1234567890", # Invalid starting digit
        "email": "priya.nair",   # Invalid email
        "address_pincode": "5600", # Invalid pincode (too short)
    }
    profile, errors = onboarding_service.onboard_taxpayer(bad_data)
    assert profile is None
    assert len(errors) >= 4
    assert any("future" in e for e in errors)
    assert any("mobile" in e for e in errors)
    assert any("email" in e for e in errors)
    assert any("PIN code" in e for e in errors)


def test_add_and_validate_bank_account(onboarding_service: TaxpayerOnboardingService):
    valid_data = {
        "pan": "ZZZPA9999Z",
        "first_name": "Ravi",
        "last_name": "Varma",
        "dob": "1991-08-25",
    }
    profile, _ = onboarding_service.onboard_taxpayer(valid_data)
    assert profile is not None

    # Invalid IFSC
    ok, errors = onboarding_service.add_bank_account(
        taxpayer_id=profile.taxpayer_id,
        account_number="1234567890",
        ifsc_code="BAD_IFSC",
        bank_name="Test Bank",
    )
    assert ok is False
    assert any("IFSC" in e for e in errors)

    # Valid bank account
    ok, errors = onboarding_service.add_bank_account(
        taxpayer_id=profile.taxpayer_id,
        account_number="112233445566",
        ifsc_code="SBIN0001234",
        bank_name="State Bank of India",
        is_refund_account=True,
    )
    assert ok is True
    assert errors == []

    updated = onboarding_service.db.get_taxpayer_profile(profile.taxpayer_id)
    assert len(updated.bank_accounts) == 1
    assert updated.bank_accounts[0].is_refund_account is True
    assert updated.bank_accounts[0].masked_account_number() == "XXXXXXXX5566"


def test_assessment_initialization(onboarding_service: TaxpayerOnboardingService):
    valid_data = {
        "pan": "ABCPE5678F",
        "first_name": "Kavita",
        "last_name": "Menon",
        "dob": "1995-12-10",
    }
    profile, _ = onboarding_service.onboard_taxpayer(valid_data)
    assert profile is not None

    # Unsupported assessment year
    assessment, errors = onboarding_service.initialize_assessment(
        taxpayer_id=profile.taxpayer_id,
        assessment_year="2020-21",
    )
    assert assessment is None
    assert any("Unsupported Assessment Year" in e for e in errors)

    # Valid assessment year (AY 2024-25)
    assessment, errors = onboarding_service.initialize_assessment(
        taxpayer_id=profile.taxpayer_id,
        assessment_year="2024-25",
        selected_regime=TaxRegime.NEW,
    )
    assert assessment is not None
    assert errors == []
    assert assessment.assessment_year == "2024-25"
    assert assessment.financial_year == "2023-24"
    assert assessment.filing_status == FilingStatus.NOT_STARTED
    assert assessment.selected_regime == TaxRegime.NEW


def test_onboarding_readiness_checklist(onboarding_service: TaxpayerOnboardingService):
    # Incomplete taxpayer profile
    partial_data = {
        "pan": "ABCPE1111F",
        "first_name": "Arjun",
        "last_name": "Das",
        "dob": "1990-05-15",
    }
    profile, _ = onboarding_service.onboard_taxpayer(partial_data)
    assessment, _ = onboarding_service.initialize_assessment(profile.taxpayer_id, "2024-25")

    status = onboarding_service.get_onboarding_status(profile.taxpayer_id, assessment.assessment_id)
    assert status["is_ready"] is False
    assert status["completion_percentage"] < 100
    assert "Bank Account" in status["missing_items"]

    # Now add bank account, address, contact
    onboarding_service.add_bank_account(
        taxpayer_id=profile.taxpayer_id,
        account_number="998877665544",
        ifsc_code="HDFC0000001",
        bank_name="HDFC Bank",
        is_refund_account=True,
    )
    profile_to_update = onboarding_service.db.get_taxpayer_profile(profile.taxpayer_id)
    profile_to_update.mobile = "9876543210"
    profile_to_update.email = "arjun@example.com"
    profile_to_update.address_city = "Chennai"
    profile_to_update.address_pincode = "600001"
    onboarding_service.db.save_taxpayer_profile(profile_to_update)

    status_complete = onboarding_service.get_onboarding_status(profile.taxpayer_id, assessment.assessment_id)
    assert status_complete["is_ready"] is True
    assert status_complete["completion_percentage"] == 100
    assert status_complete["missing_items"] == []
