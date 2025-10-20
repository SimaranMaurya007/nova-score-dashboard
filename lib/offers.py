def _interest_rate(s: float) -> float:
    if s >= 80:
        return 8.0
    if s >= 70:
        return 12.0
    if s >= 60:
        return 15.0
    return 18.0

def get_offers_for_score(score: float, worker_type: str):
    rate = _interest_rate(score)
    if worker_type == "driver":
        primary = {
            "title": "Vehicle Upgrade Loan",
            "subtitle": f"Interest {rate:.1f}% APR",
            "detail": "Upgrade or maintain your vehicle to increase earnings.",
            "cta": "Apply Now",
            "tag": "Loan",
        }
    elif worker_type == "merchant":
        primary = {
            "title": "Business Equipment Loan",
            "subtitle": f"Interest {rate:.1f}% APR",
            "detail": "Finance kitchen gear or stall upgrades to boost sales.",
            "cta": "Apply Now",
            "tag": "Loan",
        }
    else:
        primary = {
            "title": "Essential Tools Micro-Loan",
            "subtitle": f"Interest {rate:.1f}% APR",
            "detail": "Get essential tools and supplies to work more efficiently.",
            "cta": "Apply Now",
            "tag": "Loan",
        }
    return [primary]

def is_eligible_for_credit(score: float):
    return (
        score >= 70.0,
        "Eligible for credit offers" if score >= 70.0 else "Not eligible due to low Nova Score",
    )
