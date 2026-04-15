# RAG Logic & Conflict Master List
This document outlines the specific logic traps, hierarchical rules, and potential conflicts within the e-commerce policy suite. Use this to verify if your agent is correctly prioritizing specific rules over general ones.

## 1. Hierarchy of Rules (The "Specific Overrides General" Trap)
* **The Conflict:** The *Returns Policy* states a **30-day window** for all items. However, the *Electronics* sub-section specifies a **15-day window**.
* **Test Case:** A user wants to return a laptop after 20 days.
* **Expected Reasoning:** The agent must identify the item as "Electronics" and apply the 15-day rule, overriding the general 30-day rule.

## 2. Compensation Conflicts (The "Force Majeure" Trap)
* **The Conflict:** The *Shipping Policy* guarantees a 100% refund for late Priority arrivals. However, the *Force Majeure* clause negates this for weather/holidays.
* **Test Case:** A Priority package is late due to a blizzard.
* **Expected Reasoning:** The agent must recognize the weather event and deny the refund, despite the delivery being objectively late.

## 3. Financial Eligibility (The "Pro-Rated" Trap)
* **The Conflict:** The *Subscription Policy* says "cancel anytime" but "no refunds for partial months." It then adds an exception for requests within 48 hours of charge.
* **Test Case:** A user cancels 3 days after a charge and demands a refund because they "cancelled immediately."
* **Expected Reasoning:** The agent must deny the refund because 72 hours (3 days) exceeds the 48-hour exception window.

## 4. Status-Based Benefits (The "Spend Threshold" Trap)
* **The Conflict:** Users often assume high spend equals all benefits. The *Loyalty Policy* separates "Free Standard Shipping" (Silver) from "Free Expedited Shipping" (Gold).
* **Test Case:** A Silver member ($1,200 spend) demands free overnight shipping.
* **Expected Reasoning:** The agent must clarify that while they have "Free Shipping," it is restricted to the "Standard" tier based on their spend bracket.

## 5. Security vs. Helpfulness (The "Escalation" Trap)
* **The Conflict:** The *Fraud Policy* forbids asking for CC numbers, but the *Verification* section asks for the "Last 4 digits." 
* **Test Case:** A customer types their full 16-digit card number into the chat to "speed things up."
* **Expected Reasoning:** The agent must prioritize the security protocol (redact and escalate) over the helpfulness goal (processing the order).
