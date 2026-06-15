# How to Add Your Real Training Data

The model learns from examples. The more real your examples, the more useful the model becomes for YOUR business.

## What Makes Good Training Data

Each example needs 3 parts:
1. **Instruction** — What you want the model to do (e.g., "Create a CRM entry")
2. **Input** — The raw info you give it (e.g., customer details from a phone call)
3. **Output** — The exact formatted result you want back (e.g., the JSON CRM entry)

## Example Categories to Add

### 1. CRM Entries
Add 5-10 real leads you've received. Use anonymized names if you want.

### 2. Quotes
Add 3-5 actual quotes you've sent. Include your real pricing, equipment, and terms.

### 3. Invoices
Add 2-3 real invoices for completed jobs.

### 4. Emails
Add 3-5 emails you actually send: follow-ups, quote responses, welcome messages.

### 5. System Sizing
Add 2-3 real calculations you've done for customers.

### 6. Scheduling
Add 2-3 real appointment setups.

### 7. Reports
Add 1-2 weekly/monthly summary formats you use.

## Minimum Viable Dataset

| Category | Min Examples | Real Examples |
|---|---|---|
| CRM | 5 | Your actual lead format |
| Quotes | 3 | Your real pricing |
| Invoices | 2 | Your payment terms |
| Emails | 3 | Your actual tone/style |
| Scheduling | 2 | Your workflow |
| Sizing | 2 | Your calculation method |
| Reports | 1 | Your summary format |
| **TOTAL** | **18** | **~20-30 for solid results** |

## File Format

Each line in `data/solar_training_data.jsonl` is one JSON object:

```json
{"instruction": "Create a CRM entry for a residential solar lead", "input": "Name: [customer name], Phone: [number], ...", "output": "{\"action\": \"create_crm_entry\", \"data\": { ... exact format you want ... }}"}
```

## Tips

- Use your REAL company name, phone, email in outputs
- Include your ACTUAL equipment brands and prices
- Use your REAL payment terms and financing options
- Match the tone you actually use with customers
- Don't worry about perfect JSON — I can fix formatting

## Quick Start

Paste me 1-2 real examples from your business (any category) and I'll:
1. Format them correctly
2. Show you the pattern
3. Help you add the rest

