"""Independent linguistic set v1 (lexicon-locked confirmatory split).

Protocol (docs/lab/NOTEBOOK.md, 2026-08-22):
- STAGE_EMISSIONS was frozen first.
- These dialogues are rewritten from public FTC / CISA / AARP impersonation
  tropes (grandparent, fake refund, utility shutoff, romance mule, fake
  merchant, student-loan, car-warranty, Medicare plan, Amazon purchase).
- They were not used to add regexes. Vocabulary is deliberately *not*
  copied from STAGE_EMISSIONS. Some tropes will still hit the frozen
  lexicon; some will not. That is the test.
- English, single-talker, injected text (no ASR unless the caller adds it).

This set is the confirmatory linguistic population. Author-written
``vishing_scripts.heldout_scripts`` is sanity-only.
"""

from __future__ import annotations

from typing import List, Tuple

from .vishing_scripts import CallScript, _benign, _scam


def independent_scripts() -> List[CallScript]:
    return list(INDEPENDENT)


INDEPENDENT: Tuple[CallScript, ...] = (
    # ---- scam: grandparent / family emergency (FTC classic) ----
    _scam("ind_s01", "independent", "grandparent", [
        "Please don't hang up, I am calling about your grandchild.",
        "They were taken after a collision and cannot post the bond themselves.",
        "The clerk said you have a short window before they are moved.",
        "Buy three prepaid store cards at the pharmacy and read the backs.",
        "Do not loop your spouse in; they asked that this stay quiet.",
    ], trap="paraphrase"),
    _scam("ind_s02", "independent", "grandparent", [
        "This is the night intake at county holding.",
        "Your granddaughter's name is on the list for a cash appearance bond.",
        "If the amount is not at the window by lunch they stay overnight.",
        "A money transfer counter downtown is the only accepted method.",
        "Tipping off anyone else delays the release.",
    ], trap="paraphrase"),
    # ---- fake refund / treasury (not "tax bureau") ----
    _scam("ind_s03", "independent", "refund", [
        "I am with the overpayment recovery unit.",
        "A duplicate filing created a credit that will reverse unless you confirm.",
        "We need the routing digits on the account the credit hit.",
        "Then purchase a voucher so we can re-issue the correct amount.",
        "This is not public; payroll should not hear.",
    ], trap="paraphrase"),
    _scam("ind_s04", "independent", "refund", [
        "Calling from the levy resolution desk.",
        "There is a hold on a refund that will expire this morning.",
        "Confirm the last four of the identification number on file.",
        "Send a crypto voucher to the address I am about to read.",
        "Keep this off social media.",
    ], trap="paraphrase"),
    # ---- utility shutoff ----
    _scam("ind_s05", "independent", "utility", [
        "This is dispatch for the regional electric cooperative.",
        "Your meter shows a past-due balance and a disconnection order.",
        "The truck is already on the block; you have thirty minutes.",
        "Pay with a pharmacy prepaid at the number I give you.",
        "Do not ring the printed customer line; it will not stop the truck.",
    ], trap="paraphrase"),
    _scam("ind_s06", "independent", "utility", [
        "Gas service here. We have a leak flag that requires an immediate deposit.",
        "If the deposit is not posted, the valve is closed at noon.",
        "Read me the numbers on two steam cards.",
        "Your landlord should not hear this conversation.",
    ], trap="paraphrase"),
    # ---- fake merchant / Amazon-like ----
    _scam("ind_s07", "independent", "merchant", [
        "Auto-renewal desk. A four-hundred dollar device order just cleared.",
        "If that was not you, we reverse it only after a security step.",
        "I will send a one-time PIN; read it back, then the card CVV.",
        "Stay on the line until the reversal posts.",
    ], trap="paraphrase"),
    _scam("ind_s08", "independent", "merchant", [
        "Warehouse claims. A parcel in your name was flagged as high-risk.",
        "To stop a second charge, confirm the card on the account.",
        "Then we issue store credit you must claim today.",
        "Do not discuss this with the retailer app chat.",
    ], trap="paraphrase"),
    # ---- romance / mule ----
    _scam("ind_s09", "independent", "romance", [
        "It is me. The customs hold is real and I cannot board.",
        "I need you to move the sum through a transfer counter today.",
        "Do not tell your sister; she would not understand.",
        "Use bitcoin if the counter is closed.",
    ], trap="paraphrase"),
    _scam("ind_s10", "independent", "romance", [
        "The lawyer said a processing fee unlocks the crate.",
        "Western Union is the only desk they accept.",
        "Please go now; the window closes.",
        "This stays between us until I land.",
    ], trap="paraphrase"),
    # ---- student loan / debt ----
    _scam("ind_s11", "independent", "debt", [
        "I am with the forgiveness intake line.",
        "Your loans qualify if we receive a certification payment today.",
        "I need the account and routing for the small holding deposit.",
        "Missing the cutoff returns the balance to collections.",
    ], trap="paraphrase"),
    _scam("ind_s12", "independent", "debt", [
        "Auto-warranty administration. Your coverage lapses this afternoon.",
        "A technician is already assigned; we need a card to keep the slot.",
        "Read the sixteen digits and the code on the back.",
        "If you hang up the claim is void.",
    ], trap="paraphrase"),
    # ---- Medicare / health plan ----
    _scam("ind_s13", "independent", "medicare", [
        "Open-enrollment line. Your benefits change unless you opt in.",
        "I need the identification number on the red-white-blue card.",
        "There is a same-day window for the extra dental rider.",
        "Do not call the printed 800 number; it will duplicate the file.",
    ], trap="paraphrase"),
    _scam("ind_s14", "independent", "medicare", [
        "Pharmacy review. A narcotic claim in your name looks irregular.",
        "Confirm date of birth and the bank that pays the premium.",
        "A prepaid card covers the investigation fee.",
        "This is time-sensitive.",
    ], trap="paraphrase"),
    # ---- tech remote access without classic keywords ----
    _scam("ind_s15", "independent", "tech", [
        "Device security operations. We see an intrusion on your router logs.",
        "I will give you a viewer code so I can clean the worm.",
        "Leave the session open; closing it lets the process restart.",
        "Purchase a store credit so we can license the cleaner.",
    ], trap="paraphrase"),
    _scam("ind_s16", "independent", "tech", [
        "Mailbox security review. A hostile login from another country is active.",
        "Read the recovery codes in your authenticator app.",
        "Then we lock the session from here.",
        "Do not alert your employer IT; they are the suspected source.",
    ], trap="paraphrase"),
    # ---- court / legal (avoid prosecutor regex if possible) ----
    _scam("ind_s17", "independent", "court", [
        "Clerk's office, traffic division. A failure-to-appear is on the docket.",
        "A bench order prints at close of business unless a bond posts.",
        "The kiosk takes only prepaid cards.",
        "Telling your employer counts as obstruction in this county.",
    ], trap="paraphrase"),
    _scam("ind_s18", "independent", "court", [
        "This is the warrant resolution line.",
        "Marshals will detain if the amount is unpaid by tonight.",
        "Go to the transfer counter and read me the control number.",
        "Stay on the phone until I confirm with the courthouse.",
    ], trap="paraphrase"),
    # ---- lottery / prize ----
    _scam("ind_s19", "independent", "prize", [
        "Awards clearing. Your ticket matched a regional drawing.",
        "Tax on the prize is due before we can wire winnings.",
        "Pay that tax with iTunes credit I can apply from here.",
        "If you mention this, the prize is reassigned.",
    ], trap="paraphrase"),
    _scam("ind_s20", "independent", "prize", [
        "Sweepstakes fulfillment. A crate is at customs in your name.",
        "Duties must be paid this morning with a moneygram.",
        "Give me the reference they print on the receipt.",
        "Your spouse should not hear until the crate is released.",
    ], trap="paraphrase"),
    # ---- bank / account freeze (some overlap expected) ----
    _scam("ind_s21", "independent", "bank", [
        "Debit operations. We halted a wire that looks like a drain.",
        "To release the hold I need the PIN and the one-time pass from SMS.",
        "Act before the automated reversal expires.",
        "Do not call the number on the card; it will cancel my ticket.",
    ], trap="paraphrase"),
    _scam("ind_s22", "independent", "bank", [
        "Risk operations. Your mobile deposit was flagged.",
        "Confirm routing and the last paycheck amount.",
        "We will issue a replacement via crypto address for speed.",
        "Keep this confidential until the credit lands.",
    ], trap="paraphrase"),
    # ---- SSA-like without "social security" string ----
    _scam("ind_s23", "independent", "benefits", [
        "Eligibility review. A discrepancy will freeze next month's payment.",
        "Verify the account the deposit usually hits.",
        "A same-day certification fee is required on prepaid cards.",
        "Discussing this with family is treated as a tip-off.",
    ], trap="paraphrase"),
    _scam("ind_s24", "independent", "benefits", [
        "Status review unit. Your file shows an overpayment.",
        "We collect it today or the next check is reduced.",
        "Read the numbers on two gift-style pharmacy cards.",
        "Remain on the line.",
    ], trap="paraphrase"),
    # ---- benign: legitimate-sounding, some sensitive words ----
    _benign("ind_b01", "independent", "dentist", [
        "Hi, this is a reminder that you have a cleaning tomorrow at nine.",
        "No payment is due today; we have the card on file from last visit.",
        "Reply if you need to move the appointment.",
        "Have a good evening.",
    ]),
    _benign("ind_b02", "independent", "pharmacy", [
        "Your prescription is ready for pickup.",
        "The copay is the usual amount; you can pay at the counter.",
        "Bring a photo ID as we always ask.",
        "Thank you.",
    ]),
    _benign("ind_b03", "independent", "school", [
        "This is the attendance office. Your student was marked absent.",
        "Please call us back if that is an error.",
        "No action is required if you already emailed the teacher.",
    ]),
    _benign("ind_b04", "independent", "library", [
        "A hold is available at the front desk.",
        "It stays on the shelf three days.",
        "No fees unless it is late after checkout.",
    ]),
    _benign("ind_b05", "independent", "utility_legit", [
        "Monthly statement: your electric bill is ready in the portal.",
        "Autopay will run on the fifteenth as you set up last year.",
        "Ignore this if you already paid.",
    ]),
    _benign("ind_b06", "independent", "bank_legit", [
        "This is your bank. A large debit posted; if you recognize it, no action.",
        "You can review it in the app. We will never ask for your PIN on this call.",
        "Have a good day.",
    ], trap="isolated_keyword"),
    _benign("ind_b07", "independent", "irs_legit", [
        "A reminder from the tax-prep office that documents are due Friday.",
        "Bring W-2s. We do not ask for gift cards.",
        "See you then.",
    ], trap="isolated_keyword"),
    _benign("ind_b08", "independent", "ssa_legit", [
        "Your benefits statement is in the mail.",
        "No need to call. We will never demand a password.",
        "Thank you.",
    ], trap="isolated_keyword"),
    _benign("ind_b09", "independent", "retail", [
        "Your online order shipped. Tracking is in the email.",
        "The gift card you bought as a present is the SKU you chose.",
        "No further steps.",
    ], trap="isolated_keyword"),
    _benign("ind_b10", "independent", "doctor", [
        "Lab results are posted to the patient portal.",
        "The office is open until five if you have questions.",
        "This is not urgent.",
    ]),
    _benign("ind_b11", "independent", "insurer", [
        "Your claim was approved. The explanation of benefits is on the way.",
        "Nothing to pay beyond the usual deductible at the provider.",
    ]),
    _benign("ind_b12", "independent", "airline", [
        "Check-in is open for tomorrow's flight.",
        "Seat is unchanged. No payment is required.",
        "Have a nice trip.",
    ]),
    _benign("ind_b13", "independent", "survey", [
        "Would you take a three-minute survey about a recent store visit?",
        "There is no prize and no account question.",
        "You can decline.",
    ]),
    _benign("ind_b14", "independent", "password_legit", [
        "IT reminder: rotate your password this week using the self-service page.",
        "We will never ask you to read a password on the phone.",
        "Thanks.",
    ], trap="isolated_keyword"),
    _benign("ind_b15", "independent", "city", [
        "Bulk trash pickup is Tuesday. Place items at the curb after six.",
        "No fee for the first cubic yard.",
    ]),
    _benign("ind_b16", "independent", "charity", [
        "This is a registered food-bank fundraiser.",
        "Any gift is optional and goes through our public website.",
        "We do not take wire transfers on this call.",
    ]),
)
