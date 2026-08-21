"""Labeled English call scripts for linguistic evaluation.

Train vs held-out is explicit. Held-out scam lines are paraphrases that
avoid the high-precision keyword groups where possible, so a bag-of-words
keyword detector can be compared with the stage tracker.

Benign traps contain isolated sensitive words (bank, password, gift card
as retail, verify an email) without a vishing trajectory.

These are author-written synthetic scripts, not transcripts of real
victims. They are for method evaluation, not prevalence estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


Turn = Tuple[float, str]


@dataclass(frozen=True)
class CallScript:
    script_id: str
    split: str  # train | heldout
    family: str
    is_scam: bool
    turns: Tuple[Turn, ...]
    trap: str = ""  # isolated_keyword | paraphrase | canonical | none


def _t(*lines: str, dt: float = 0.8) -> Tuple[Turn, ...]:
    return tuple((i * dt, line) for i, line in enumerate(lines))


def _scam(sid: str, split: str, family: str, lines: Sequence[str], trap: str = "canonical") -> CallScript:
    return CallScript(sid, split, family, True, _t(*lines), trap)


def _benign(sid: str, split: str, family: str, lines: Sequence[str], trap: str = "none") -> CallScript:
    return CallScript(sid, split, family, False, _t(*lines), trap)


SCRIPTS: Tuple[CallScript, ...] = (
    # ---- train scam: canonical high-precision phrases ----
    _scam("s01", "train", "irs", [
        "Hello, this is a courtesy call from the IRS.",
        "We detected unusual activity on your tax file.",
        "You must act immediately within 2 hours.",
        "Verify your social security number now.",
        "Purchase gift cards and read the numbers.",
        "Do not tell anyone or there is a warrant for your arrest.",
    ]),
    _scam("s02", "train", "ssa", [
        "Good morning, calling from social security.",
        "Your benefits are suspended due to suspicious activity.",
        "You need to verify your date of birth and bank account.",
        "Send money by wire transfer right now.",
        "Keep this confidential.",
        "Legal action will follow if you refuse.",
    ]),
    _scam("s03", "train", "bank", [
        "This is a call from your bank fraud department.",
        "Your account is compromised.",
        "Act now before it is locked.",
        "Give me your routing number and PIN.",
        "Buy iTunes cards to secure the account.",
        "Stay on the line and do not call the bank.",
    ]),
    _scam("s04", "train", "tech_support", [
        "Hello from Microsoft support.",
        "We found a virus and malware on your computer.",
        "You must grant remote access through TeamViewer immediately.",
        "Your password has been stolen.",
        "Pay with a bitcoin voucher.",
        "Do not inform anyone.",
    ]),
    _scam("s05", "train", "family", [
        "Please don't hang up, this is about your grandson.",
        "He was in an accident and is in the hospital.",
        "He has been arrested and needs bail right now.",
        "Wire transfer the money through Western Union.",
        "Keep this between us.",
        "If you tell your family he will stay in jail.",
    ]),
    _scam("s06", "train", "medicare", [
        "Calling from Medicare about your card.",
        "The account is locked because of fraud.",
        "You have to verify the card number and CVV today.",
        "There is a one-time password we need.",
        "Do not tell your doctor.",
        "You will lose your benefits otherwise.",
    ]),
    _scam("s07", "train", "apple", [
        "This is Apple support regarding your iCloud.",
        "Unusual activity was flagged as fraudulent.",
        "Confirm your verification code immediately.",
        "Purchase a steam card for the unlock fee.",
        "Stay on the phone.",
        "We can take legal action.",
    ]),
    _scam("s08", "train", "doj", [
        "This is a federal agent with the department of justice.",
        "There is a warrant linked to your social security number.",
        "You must pay a fine with MoneyGram now.",
        "Give me your bank account to process the hold.",
        "This is secret.",
        "You will be arrested this afternoon.",
    ]),
    _scam("s09", "train", "irs", [
        "Internal revenue calling about unpaid tax.",
        "Your file is suspended.",
        "Act within 60 minutes.",
        "Read me the SSN and mother's maiden name.",
        "Send cryptocurrency to the wallet we provide.",
        "Don't tell your accountant.",
    ]),
    _scam("s10", "train", "bank", [
        "Hello from law enforcement working with your bank.",
        "There is a fraudulent wire on the account.",
        "You need to verify the card number right now.",
        "Buy gift cards at the store and stay on the line.",
        "Do not call anyone.",
        "A warrant is already filed.",
    ]),
    _scam("s11", "train", "tech_support", [
        "Apple support here, your computer is infected with a virus.",
        "Malware is stealing the password.",
        "Install AnyDesk immediately.",
        "Pay the removal fee with bitcoin.",
        "Keep this confidential.",
        "You will be fined if you hang up.",
    ]),
    _scam("s12", "train", "family", [
        "Your granddaughter is in trouble after an accident.",
        "She is at the hospital and needs bail.",
        "Send money through Western Union now.",
        "Don't tell her mother.",
        "If you delay she stays jailed.",
    ]),
    _scam("s13", "train", "ssa", [
        "Social security administration courtesy call.",
        "Benefits are blocked due to unusual activity.",
        "Verify date of birth and routing number.",
        "Wire transfer a processing fee.",
        "Do not inform the local office.",
        "You will be deported or lose benefits.",
    ]),
    _scam("s14", "train", "utility", [
        "This is a call from the electric company legal department.",
        "Service is compromised and will be locked today.",
        "You must pay immediately with a gift card.",
        "Read the PIN on the card.",
        "Stay on the line.",
        "A warrant follows nonpayment.",
    ]),
    _scam("s15", "train", "amazon", [
        "Calling from your bank about an Amazon refund.",
        "The order is fraudulent.",
        "Verify CVV and one-time password now.",
        "We will send a bitcoin refund after you pay a fee.",
        "Do not tell anyone.",
        "Legal action is pending.",
    ]),
    _scam("s16", "train", "irs", [
        "Federal agent, IRS division.",
        "Your social security was used in a crime.",
        "You must act now.",
        "Purchase gift cards for the settlement.",
        "Keep this secret.",
        "Arrest is scheduled.",
    ]),
    _scam("s17", "train", "tech_support", [
        "Microsoft support, we detected malware.",
        "Your computer was breached.",
        "Give remote access through TeamViewer.",
        "Pay with iTunes cards immediately.",
        "Do not call anyone else.",
    ]),
    _scam("s18", "train", "bank", [
        "Hello, this is a courtesy call from the fraud unit at your bank.",
        "Unusual activity, account compromised.",
        "Verify SSN and PIN within 2 hours.",
        "Wire transfer the disputed funds to a safe account.",
        "Between us, do not walk into a branch.",
        "You will be prosecuted if you refuse.",
    ]),
    _scam("s19", "train", "lottery", [
        "Congratulations from the federal prize office.",
        "Your claim is locked until we verify identity.",
        "Give bank account and date of birth now.",
        "Pay the tax with a MoneyGram.",
        "Do not tell friends.",
        "The prize expires in 30 minutes.",
    ]),
    _scam("s20", "train", "immigration", [
        "Department of justice immigration unit.",
        "There is a warrant and you may be deported.",
        "Pay the fine with cryptocurrency immediately.",
        "Verify social security number.",
        "Stay on the phone, this is confidential.",
        "Jail time if you hang up.",
    ]),
    # ---- held-out scam: paraphrases / broader stage language ----
    _scam("h01", "heldout", "tax_office", [
        "Good afternoon, I am with the national tax bureau.",
        "Your filing shows a discrepancy that froze the file.",
        "This has to be cleared before noon.",
        "I need the taxpayer identification you use on wages.",
        "Go to the pharmacy and get prepaid store cards, then read the back.",
        "If this leaks to a relative, marshals will detain you today.",
    ], "paraphrase"),
    _scam("h02", "heldout", "benefits", [
        "I am calling from the benefits integrity desk.",
        "Payments are frozen after a mismatch on the file.",
        "We have a short window this morning.",
        "Confirm the number on your wage statements and the account you use for deposits.",
        "A prepaid card is the only accepted clearing method.",
        "Do not discuss this with neighbors; officers are already assigned.",
    ], "paraphrase"),
    _scam("h03", "heldout", "card_services", [
        "This is card services calling about a hold on the plastic you carry.",
        "Someone tried to drain the account overnight.",
        "We cannot wait until tomorrow.",
        "Read the long number and the three digits on the back.",
        "Then buy prepaid store cards so we can reverse the hold.",
        "Remain on this handset and do not ring the number on the card.",
    ], "paraphrase"),
    _scam("h04", "heldout", "helpdesk", [
        "Helpdesk here, your device is flagged for a remote intrusion.",
        "A worm is sitting on the machine.",
        "Share the screen through the remote tool I name in a moment.",
        "A crypto voucher covers the cleanup fee.",
        "This conversation is not for your spouse.",
        "Disconnecting is treated as obstruction.",
    ], "paraphrase"),
    _scam("h05", "heldout", "relative", [
        "I am a public defender calling about a young relative of yours.",
        "There was a crash and he is being held downtown.",
        "Bail has to move in the next hour.",
        "Use a money transfer counter, not a bank teller.",
        "If the family group chat hears this, the hold gets worse.",
        "They will keep him overnight.",
    ], "paraphrase"),
    _scam("h06", "heldout", "health_plan", [
        "This is the enrollment desk for your health plan.",
        "The membership looks tampered with.",
        "We need the identifier from your card before close of business.",
        "A prepaid card pays the reactivation levy.",
        "Do not mention this visit to your clinic.",
        "Coverage will lapse and collections will follow.",
    ], "paraphrase"),
    _scam("h07", "heldout", "cloud_desk", [
        "Cloud account desk, your login from another country looks hostile.",
        "The mailbox is on a freeze.",
        "I will send a numeric passkey, read it back, then we need store credit codes.",
        "Remain in this session.",
        "A complaint has already been drafted.",
    ], "paraphrase"),
    _scam("h08", "heldout", "prosecutor", [
        "I work with a prosecutor's intake line.",
        "Your wage number appears on a sealed complaint.",
        "A same-day settlement is available.",
        "Prepaid cards or a crypto address are listed on the form.",
        "This filing is not public yet.",
        "If you skip the settlement, pickup is this evening.",
    ], "paraphrase"),
    _scam("h09", "heldout", "utility_desk", [
        "Dispatch from the power cooperative.",
        "A technician is queued to cut the meter for unpaid charges that look forged.",
        "You can stop the truck if you clear the amount with store credit today.",
        "Read the codes from the cards.",
        "Do not call the number printed on last month's bill while we are working it.",
        "Nonpayment is being sent to the courthouse.",
    ], "paraphrase"),
    _scam("h10", "heldout", "parcel", [
        "Parcel claims unit. A delivery in your name was used in a fraud ring.",
        "The case is time-sensitive this morning.",
        "Confirm the digits on the card that paid the label.",
        "We also need prepaid cards to release the hold.",
        "Keep this off social media.",
        "Refusal is logged as non-cooperation.",
    ], "paraphrase"),
    _scam("h11", "heldout", "tax_office", [
        "National assessment office on the line.",
        "A second filing collided with yours and triggered a freeze.",
        "Clearance is only this morning.",
        "I need the identifier from your wage slips.",
        "Pharmacy prepaid cards, then the numbers on the reverse.",
        "Talking to a neighbor about this is treated as tipping off.",
    ], "paraphrase"),
    _scam("h12", "heldout", "helpdesk", [
        "Device security desk. Intrusion signature on your router.",
        "A remote session is required to quarantine it.",
        "Cleanup is billed as store credit, not a card on file.",
        "Your partner should not hear the passcodes.",
        "Closing the session counts as declining the quarantine.",
    ], "paraphrase"),
    _scam("h13", "heldout", "relative", [
        "Court clerk calling. A younger family member missed a hearing after a crash.",
        "A bond has to post within the hour.",
        "Use a transfer counter across town.",
        "Do not loop in the rest of the family yet.",
        "Otherwise he stays in holding.",
    ], "paraphrase"),
    _scam("h14", "heldout", "benefits", [
        "Integrity review for monthly payments.",
        "A mismatch froze the next deposit.",
        "Window closes at midday.",
        "Confirm wage identifier and deposit account.",
        "Prepaid card is the recovery rail.",
        "Speaking with the local office while this is open will delay you further.",
    ], "paraphrase"),
    _scam("h15", "heldout", "card_services", [
        "Overnight drain attempt on the account.",
        "We can reverse it only if you stay with me.",
        "Long number, back digits, then prepaid cards for the reversal escrow.",
        "Do not ring the printed service line; it is saturated.",
        "The case is already with investigations.",
    ], "paraphrase"),
    _scam("h16", "heldout", "immigration_desk", [
        "Intake for status review.",
        "A pickup order is in draft over a wage-number collision.",
        "Same-day clearing is a crypto transfer.",
        "Confirm the wage identifier.",
        "This draft is not for your employer.",
        "Walking away leaves the pickup on the calendar.",
    ], "paraphrase"),
    _scam("h17", "heldout", "prize_desk", [
        "Awards clearing house. Your claim is parked pending identity.",
        "We need deposit coordinates this morning.",
        "Tax on the award is prepaid cards at a counter.",
        "Friends cannot know until it posts.",
        "The claim window is thirty minutes.",
    ], "paraphrase"),
    _scam("h18", "heldout", "prosecutor", [
        "Intake line again with a sealed matter on your wage number.",
        "Settlement today avoids the pickup.",
        "Store credit codes or a crypto address as listed.",
        "File is not public.",
        "Skip it and the evening unit has your address.",
    ], "paraphrase"),
    _scam("h19", "heldout", "health_plan", [
        "Membership looks altered.",
        "Identifier from the card before we close.",
        "Reactivation levy on a prepaid card.",
        "Clinic staff should not be copied.",
        "Lapse plus collections if you pass.",
    ], "paraphrase"),
    _scam("h20", "heldout", "cloud_desk", [
        "Foreign login froze the mailbox.",
        "Read back the passkey, then store credit to lift the freeze.",
        "Stay in session.",
        "A complaint draft is already in the folder.",
    ], "paraphrase"),
    # ---- train benign ----
    _benign("b01", "train", "dentist", [
        "Hello, this is a reminder about your dentist appointment tomorrow.",
        "Please arrive ten minutes early and bring your insurance card.",
        "Thank you and have a nice day.",
    ]),
    _benign("b02", "train", "school", [
        "Good morning, calling from the elementary school office.",
        "The play starts at six, please bring a snack to share.",
        "Have a good afternoon.",
    ]),
    _benign("b03", "train", "pharmacy", [
        "Your prescription is ready for pickup.",
        "The pharmacy closes at seven.",
        "Thank you.",
    ]),
    _benign("b04", "train", "survey", [
        "We are running a short customer survey about your last visit.",
        "There is no payment requested.",
        "Have a nice day.",
    ]),
    _benign("b05", "train", "library", [
        "The book you reserved is on the hold shelf.",
        "Please come by this week.",
        "Thank you.",
    ]),
    _benign("b06", "train", "clinic", [
        "Reminder from the clinic about your physical next Tuesday.",
        "Bring a list of current medicines.",
        "See you then.",
    ]),
    _benign("b07", "train", "airline", [
        "Your flight is on time, gate change to B12.",
        "Have a good trip.",
    ]),
    _benign("b08", "train", "weather", [
        "This is an automated weather alert for your county.",
        "Rain is expected tonight.",
        "Thank you for listening.",
    ]),
    _benign("b09", "train", "volunteer", [
        "Calling about Saturday's park cleanup.",
        "Please bring gloves if you have them.",
        "Hope to see you there.",
    ]),
    _benign("b10", "train", "restaurant", [
        "Confirming your dinner reservation for two at seven.",
        "We look forward to hosting you.",
        "Goodbye.",
    ]),
    _benign("b11", "train", "delivery", [
        "Your package will arrive this afternoon.",
        "No signature is required.",
        "Have a nice day.",
    ]),
    _benign("b12", "train", "gym", [
        "The gym will open an hour later on Monday.",
        "Thank you for being a member.",
    ]),
    _benign("b13", "train", "vet", [
        "Reminder that the dog's checkup is Friday.",
        "Please bring the vaccination booklet.",
        "See you soon.",
    ]),
    _benign("b14", "train", "pta", [
        "PTA meeting is moved to the library.",
        "Coffee will be provided.",
        "Thanks.",
    ]),
    _benign("b15", "train", "utilities_ok", [
        "Scheduled maintenance will pause water from two to four.",
        "You do not need to call us back.",
        "Thank you.",
    ]),
    _benign("b16", "train", "hr", [
        "Human resources confirming the all-hands on Wednesday.",
        "The agenda is in your inbox.",
        "Have a good day.",
    ]),
    _benign("b17", "train", "church", [
        "Choir practice is cancelled this evening.",
        "We will meet next week.",
        "Take care.",
    ]),
    _benign("b18", "train", "insurance_ok", [
        "Your policy documents are in the mail.",
        "No action is needed.",
        "Thank you.",
    ]),
    _benign("b19", "train", "hotel", [
        "Your hotel check-in is after three.",
        "Breakfast is included.",
        "See you soon.",
    ]),
    _benign("b20", "train", "class", [
        "The evening class will use room 204 tonight.",
        "Please bring a notebook.",
        "Thanks.",
    ]),
    # isolated-keyword benign traps (train)
    _benign("t01", "train", "trap_bank", [
        "The bank downtown will be closed Monday for a holiday.",
        "Use the app for a balance if you need it.",
        "Thank you.",
    ], "isolated_keyword"),
    _benign("t02", "train", "trap_password", [
        "Your password reset for the portal was successful.",
        "If you did not request it, use the website help page.",
        "Have a nice day.",
    ], "isolated_keyword"),
    _benign("t03", "train", "trap_gift", [
        "Gift cards are ten percent off at the grocery store this weekend.",
        "It is a regular promotion.",
        "Thank you for shopping.",
    ], "isolated_keyword"),
    _benign("t04", "train", "trap_verify", [
        "Please verify you received the appointment email we sent.",
        "Reply to the clinic if the time does not work.",
        "See you tomorrow.",
    ], "isolated_keyword"),
    _benign("t05", "train", "trap_ssn_office", [
        "The social security office on Main Street has new Saturday hours.",
        "This is an informational recording.",
        "Goodbye.",
    ], "isolated_keyword"),
    _benign("t06", "train", "trap_account", [
        "Your bank account statement is available in online banking.",
        "No phone verification is required.",
        "Thank you.",
    ], "isolated_keyword"),
    _benign("t07", "train", "trap_irs_site", [
        "The IRS website lists updated tax deadlines this year.",
        "We are the library reminding you of the public computer class.",
        "Have a good afternoon.",
    ], "isolated_keyword"),
    _benign("t08", "train", "trap_arrest_news", [
        "This is the local news desk asking if you saw the parade.",
        "There is no warrant and no payment.",
        "Thank you for your time.",
    ], "isolated_keyword"),
    # ---- held-out benign ----
    _benign("hb01", "heldout", "optometrist", [
        "Hello from the vision clinic, your glasses are ready.",
        "Parking is behind the building.",
        "Have a good day.",
    ]),
    _benign("hb02", "heldout", "soccer", [
        "Practice is moved to the indoor field because of rain.",
        "Bring indoor shoes.",
        "Thanks coaches.",
    ]),
    _benign("hb03", "heldout", "museum", [
        "Your timed ticket is for three o'clock.",
        "Please do not arrive more than fifteen minutes early.",
        "Enjoy the exhibit.",
    ]),
    _benign("hb04", "heldout", "council", [
        "City council reminder: recycling pickup is delayed one day.",
        "No need to return this call.",
        "Thank you.",
    ]),
    _benign("hb05", "heldout", "tutor", [
        "Tutoring is still on for Thursday.",
        "Please bring last week's worksheet.",
        "See you then.",
    ]),
    _benign("hb06", "heldout", "bakery", [
        "The cake you ordered will be ready at noon.",
        "Pay at the counter as usual.",
        "Bye.",
    ]),
    _benign("hb07", "heldout", "union", [
        "Union meeting is in the cafeteria.",
        "Agenda is posted on the board.",
        "Thank you.",
    ]),
    _benign("hb08", "heldout", "bus", [
        "School bus route 12 is five minutes late.",
        "Drivers were notified.",
        "Have a good morning.",
    ]),
    _benign("hb09", "heldout", "alumni", [
        "Alumni weekend registration is open online.",
        "No payment over the phone.",
        "Hope you can join.",
    ]),
    _benign("hb10", "heldout", "garden", [
        "Community garden shift is Saturday at nine.",
        "Tools are provided.",
        "Thanks.",
    ]),
    _benign("ht01", "heldout", "trap_bank", [
        "Your bank's mobile deposit tutorial is on their website.",
        "This is the adult-education center sharing a class link.",
        "Have a nice evening.",
    ], "isolated_keyword"),
    _benign("ht02", "heldout", "trap_gift", [
        "The bookstore is selling gift cards for Teacher Appreciation week.",
        "Stop by if you want one, no rush.",
        "Thank you.",
    ], "isolated_keyword"),
    _benign("ht03", "heldout", "trap_verify", [
        "Please verify the spelling of your name on the conference badge email.",
        "Reply to the organizer if it is wrong.",
        "See you next month.",
    ], "isolated_keyword"),
    _benign("ht04", "heldout", "trap_password", [
        "IT reminds everyone that password changes happen on the first Monday.",
        "Use the official portal, not the phone.",
        "Thank you.",
    ], "isolated_keyword"),
    _benign("ht05", "heldout", "trap_medicare", [
        "The senior center is hosting a talk about Medicare open enrollment.",
        "It is educational, not a billing call.",
        "Coffee is provided.",
    ], "isolated_keyword"),
    _benign("ht06", "heldout", "trap_account", [
        "Your utility account portal has a new statement view.",
        "No action required on this call.",
        "Goodbye.",
    ], "isolated_keyword"),
    _benign("ht07", "heldout", "trap_federal", [
        "The federal holiday calendar is posted at city hall.",
        "Offices are closed Thursday.",
        "Thank you.",
    ], "isolated_keyword"),
    _benign("ht08", "heldout", "trap_arrest_play", [
        "Tickets are on sale for the community play about a wrongful arrest.",
        "It is theater, not a legal notice.",
        "Have a good night.",
    ], "isolated_keyword"),
)


def all_scripts() -> List[CallScript]:
    return list(SCRIPTS)


def train_scripts() -> List[CallScript]:
    return [s for s in SCRIPTS if s.split == "train"]


def heldout_scripts() -> List[CallScript]:
    return [s for s in SCRIPTS if s.split == "heldout"]
