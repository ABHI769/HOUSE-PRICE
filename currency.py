"""Currency conversion utilities. Base currency for the model is INR."""

EXCHANGE_RATES = {
    "INR": 1.0,
    "USD": 0.012,
    "EUR": 0.011,
    "GBP": 0.0095,
    "AED": 0.044,
    "SGD": 0.016,
    "AUD": 0.018,
    "CAD": 0.016,
    "JPY": 1.82,
    "CNY": 0.087,
}

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "AED": "د.إ",
    "SGD": "S$",
    "AUD": "A$",
    "CAD": "C$",
    "JPY": "¥",
    "CNY": "¥",
}


def convert_from_inr(amount_inr: float, target_currency: str) -> float:
    currency = target_currency.upper()
    if currency not in EXCHANGE_RATES:
        raise ValueError(f"Unsupported currency: {target_currency}")
    return amount_inr * EXCHANGE_RATES[currency]


def format_price(amount: float, currency: str) -> str:
    currency = currency.upper()
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    if currency == "INR":
        return f"{symbol}{amount:,.0f}"
    if currency == "JPY":
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def get_supported_currencies():
    return [
        {"code": code, "symbol": CURRENCY_SYMBOLS[code], "name": _currency_name(code)}
        for code in EXCHANGE_RATES
    ]


def _currency_name(code: str) -> str:
    names = {
        "INR": "Indian Rupee",
        "USD": "US Dollar",
        "EUR": "Euro",
        "GBP": "British Pound",
        "AED": "UAE Dirham",
        "SGD": "Singapore Dollar",
        "AUD": "Australian Dollar",
        "CAD": "Canadian Dollar",
        "JPY": "Japanese Yen",
        "CNY": "Chinese Yuan",
    }
    return names.get(code, code)
