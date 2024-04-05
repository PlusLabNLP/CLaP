patterns = {
    "ace05": {
        "Business:Declare-Bankruptcy": {
            "event subtype": "declare bankruptcy",
            "event type": "Business:Declare-Bankruptcy",
            "event description": "The event is related to some organization declaring bankruptcy.",
            "valid roles": ["Org", "Place"],
        },
        "Business:End-Org": {
            "event subtype": "end organization",
            "event type": "Business:End-Org",
            "event description": "The event is related to some organization ceasing to exist.",
            "valid roles": ["Org", "Place"],
        },
        "Business:Merge-Org": {
            "event subtype": "merge organization",
            "event type": "Business:Merge-Org",
            "event description": "The event is related to two or more organization coming together to form a new organization.",
            "valid roles": ["Org", "Place"],
        },
        "Business:Start-Org": {
            "event subtype": "start organization",
            "event type": "Business:Start-Org",
            "event description": "The event is related to a new organization being created.",
            "valid roles": ["Agent", "Org", "Place"],
        },
        "Conflict:Attack": {
            "event subtype": "attack",
            "event type": "Conflict:Attack",
            "event description": "The event is related to conflict and some violent physical act.",
            "valid roles": ["Attacker", "Target", "Instrument", "Place"],
        },
        "Conflict:Demonstrate": {
            "event subtype": "demonstrate",
            "event type": "Conflict:Demonstrate",
            "event description": "The event is related to a large number of people coming together to protest.",
            "valid roles": ["Entity", "Place"],
        },
        "Contact:Meet": {
            "event subtype": "meet",
            "event type": "Contact:Meet",
            "event description": "The event is related to a group of people meeting and interacting with one another face-to-face.",
            "valid roles": ["Entity", "Place"],
        },
        "Contact:Phone-Write": {
            "event subtype": "phone write",
            "event type": "Contact:Phone-Write",
            "event description": "The event is related to people phone calling or messaging one another.",
            "valid roles": ["Entity", "Place"],
        },
        "Justice:Acquit": {
            "event subtype": "acquit",
            "event type": "Justice:Acquit",
            "event description": "The event is related to someone being acquitted.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        },
        "Justice:Appeal": {
            "event subtype": "appeal",
            "event type": "Justice:Appeal",
            "event description": "The event is related to someone appealing the decision of a court.",
            "valid roles": ["Plaintiff", "Place", "Adjudicator"],
        },
        "Justice:Arrest-Jail": {
            "event subtype": "arrest jail",
            "event type": "Justice:Arrest-Jail",
            "event description": "The event is related to a person getting arrested or a person being sent to jail.",
            "valid roles": ["Person", "Agent", "Place"],
        },
        "Justice:Charge-Indict": {
            "event subtype": "charge indict",
            "event type": "Justice:Charge-Indict",
            "event description": "The event is related to someone or some organization being accused of a crime.",
            "valid roles": ["Defendant", "Prosecutor", "Place", "Adjudicator"],
        }, 
        "Justice:Convict": {
            "event subtype": "convict",
            "event type": "Justice:Convict",
            "event description": "The event is related to someone being found guilty of a crime.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        }, 
        "Justice:Execute": {
            "event subtype": "execute",
            "event type": "Justice:Execute",
            "event description": "The event is related to someone being executed to death.",
            "valid roles": ["Person", "Agent", "Place"],
        },
        "Justice:Extradite": {
            "event subtype": "extradite",
            "event type": "Justice:Extradite",
            "event description": "The event is related to justice. The event occurs when a person was extradited from one place to another place.",
            "valid roles": ["Person", "Destination", "Origin", "Agent"],
        },
        "Justice:Fine": {
            "event subtype": "fine",
            "event type": "Justice:Fine",
            "event description": "The event is related to someone being issued a financial punishment.",
            "valid roles": ["Entity", "Place", "Adjudicator"], 
        },
        "Justice:Pardon": {
            "event subtype": "pardon",
            "event type": "Justice:Pardon",
            "event description": "The event is related to someone being pardoned.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        }, 
        "Justice:Release-Parole": {
            "event subtype": "release parole",
            "event type": "Justice:Release-Parole",
            "event description": "The event is related to an end to someone's custody in prison.",
            "valid roles": ["Person", "Entity", "Place", "Adjudicator", "Defendant"],
        },
        "Justice:Sentence": {
            "event subtype": "sentence",
            "event type": "Justice:Sentence",
            "event description": "The event is related to someone being sentenced to punishment because of a crime.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        }, 
        "Justice:Sue": {
            "event subtype": "sue",
            "event type": "Justice:Sue",
            "event description": "The event is related to a court proceeding that has been initiated and someone sue the other.",
            "valid roles": ["Defendant", "Plaintiff", "Place", "Adjudicator"], 
        }, 
        "Justice:Trial-Hearing": {
            "event subtype": "trial hearing",
            "event type": "Justice:Trial-Hearing",
            "event description": "The event is related to a trial or hearing for someone.",
            "valid roles": ["Defendant", "Prosecutor", "Place", "Adjudicator"], 
        }, 
        "Life:Be-Born": {
            "event subtype": "born",
            "event type": "Life:Be-Born",
            "event description": "The event is related to life and someone is given birth to.",
            "valid roles": ["Person", "Place"], 
        }, 
        "Life:Die": {
            "event subtype": "die",
            "event type": "Life:Die",
            "event description": "The event is related to life and someone died.",
            "valid roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Life:Divorce": {
            "event subtype": "divorce",
            "event type": "Life:Divorce",
            "event description": "The event is related to life and someone was divorced.",
            "valid roles": ["Person", "Place"], 
        }, 
        "Life:Injure": {
            "event subtype": "injure",
            "event type": "Life:Injure",
            "event description": "The event is related to life and someone is injured.",
            "valid roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Life:Marry": {
            "event subtype": "marry",
            "event type": "Life:Marry",
            "event description": "The event is related to life and someone is married.",
            "valid roles": ["Person", "Place"], 
        },
        "Movement:Transport": {
            "event subtype": "transport",
            "event type": "Movement:Transport",
            "event description": "The event is related to movement. The event occurs when a weapon or vehicle is moved from one place to another.",
            "valid roles": ["Artifact", "Destination", "Origin", "Vehicle", "Agent", "Place"], 
        },
        "Personnel:Elect": {
            "event subtype": "elect",
            "event type": "Personnel:Elect",
            "event description": "The event is related to a candidate wins an election.",
            "valid roles": ["Person", "Entity", "Place"], 
        },
        "Personnel:End-Position": {
            "event subtype": "end position",
            "event type": "Personnel:End-Position",
            "event description": "The event is related to a person stops working for an organization or a hiring manager.",
            "valid roles": ["Person", "Entity", "Place"],
        }, 
        "Personnel:Nominate": {
            "event subtype": "nominate",
            "event type": "Personnel:Nominate",
            "event description": "The event is related to a person being nominated for a position.",
            "valid roles": ["Person", "Agent", "Place"], 
        },  
        "Personnel:Start-Position": {
            "event subtype": "start position",
            "event type": "Personnel:Start-Position",
            "event description": "The event is related to a person begins working for an organization or a hiring manager.",
            "valid roles": ["Person", "Entity", "Place"], 
        },  
        "Transaction:Transfer-Money": {
            "event subtype": "transfer money",
            "event type": "Transaction:Transfer-Money",
            "event description": "The event is related to transaction. The event occurs when someone is giving, receiving, borrowing, or lending money.",
            "valid roles": ["Giver", "Recipient", "Place", "Beneficiary"], 
        },  
        "Transaction:Transfer-Ownership": {
            "event subtype": "transfer ownership",
            "event type": "Transaction:Transfer-Ownership",
            "event description": "The event is related to transaction. The event occurs when an item or an organization is sold or gave to some other.",
            "valid roles": ["Buyer", "Artifact", "Seller", "Place", "Beneficiary"], 
        }, 
    }, 
    "ere": {
        "Business:Declare-Bankruptcy": {
            "event subtype": "declare bankruptcy",
            "event type": "Business:Declare-Bankruptcy",
            "event description": "The event is related to some organization declaring bankruptcy.",
            "valid roles": ["Org", "Place"],
        },
        "Business:End-Org": {
            "event subtype": "end organization",
            "event type": "Business:End-Org",
            "event description": "The event is related to some organization ceasing to exist.",
            "valid roles": ["Org", "Place"],
        },
        "Business:Merge-Org": {
            "event subtype": "merge organization",
            "event type": "Business:Merge-Org",
            "event description": "The event is related to two or more organization coming together to form a new organization.",
            "valid roles": ["Org"],
        },
        "Business:Start-Org": {
            "event subtype": "start organization",
            "event type": "Business:Start-Org",
            "event description": "The event is related to a new organization being created.",
            "valid roles": ["Agent", "Org", "Place"],
        },
        "Conflict:Attack": {
            "event subtype": "attack",
            "event type": "Conflict:Attack",
            "event description": "The event is related to conflict and some violent physical act.",
            "valid roles": ["Attacker", "Target", "Instrument", "Place"],
        },
        "Conflict:Demonstrate": {
            "event subtype": "demonstrate",
            "event type": "Conflict:Demonstrate",
            "event description": "The event is related to a large number of people coming together to protest.",
            "valid roles": ["Entity", "Place"],
        },
        "Contact:Broadcast": {
            "event subtype": "broadcast",
            "event type": "Contact:Broadcast",
            "event description": "The event happens when a person or an organization contact with the media and other publicity or announcement conference.",
            "valid roles": ["Entity", "Place", "Audience"],
        },
        "Contact:Contact": {
            "event subtype": "contact",
            "event type": "Contact:Contact",
            "event description": "The event happens when there is no explicit mention of contact ways of communication.",
            "valid roles": ["Entity", "Place"],
        },
        "Contact:Correspondence": {
            "event subtype": "contact correspondence",
            "event type": "Contact:Correspondence",
            "event description": "The event happens when a face‐to‐face meeting between sender and receiver is not explicitly stated. This includes written, phone, or electronic communication.",
            "valid roles": ["Entity", "Place"],
        },
        "Contact:Meet": {
            "event subtype": "meet",
            "event type": "Contact:Meet",
            "event description": "The event is related to a group of people meeting and interacting with one another face-to-face.",
            "valid roles": ["Entity", "Place"],
        },
        "Justice:Acquit": {
            "event subtype": "acquit",
            "event type": "Justice:Acquit",
            "event description": "The event is related to someone being acquitted.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        },
        "Justice:Appeal": {
            "event subtype": "appeal",
            "event type": "Justice:Appeal",
            "event description": "The event is related to someone appealing the decision of a court.",
            "valid roles": ["Defendant", "Prosecutor", "Place", "Adjudicator"],
        },
        "Justice:Arrest-Jail": {
            "event subtype": "arrest jail",
            "event type": "Justice:Arrest-Jail",
            "event description": "The event is related to a person getting arrested or a person being sent to jail.",
            "valid roles": ["Person", "Agent", "Place"],
        },
        "Justice:Charge-Indict": {
            "event subtype": "charge indict",
            "event type": "Justice:Charge-Indict",
            "event description": "The event is related to someone or some organization being accused of a crime.",
            "valid roles": ["Defendant", "Prosecutor", "Place", "Adjudicator"],
        }, 
        "Justice:Convict": {
            "event subtype": "convict",
            "event type": "Justice:Convict",
            "event description": "The event is related to someone being found guilty of a crime.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        }, 
        "Justice:Execute": {
            "event subtype": "execute",
            "event type": "Justice:Execute",
            "event description": "The event is related to someone being executed to death.",
            "valid roles": ["Person", "Agent", "Place"],
        },
        "Justice:Extradite": {
            "event subtype": "extradite",
            "event type": "Justice:Extradite",
            "event description": "The event is related to justice. The event occurs when a person was extradited from one place to another place.",
            "valid roles": ["Person", "Destination", "Origin", "Agent"],
        },
        "Justice:Fine": {
            "event subtype": "fine",
            "event type": "Justice:Fine",
            "event description": "The event is related to someone being issued a financial punishment.",
            "valid roles": ["Entity", "Place", "Adjudicator"], 
        },
        "Justice:Pardon": {
            "event subtype": "pardon",
            "event type": "Justice:Pardon",
            "event description": "The event is related to someone being pardoned.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        }, 
        "Justice:Release-Parole": {
            "event subtype": "release parole",
            "event type": "Justice:Release-Parole",
            "event description": "The event is related to an end to someone's custody in prison.",
            "valid roles": ["Person", "Agent", "Place"],
        },
        "Justice:Sentence": {
            "event subtype": "sentence",
            "event type": "Justice:Sentence",
            "event description": "The event is related to someone being sentenced to punishment because of a crime.",
            "valid roles": ["Defendant", "Place", "Adjudicator"],
        }, 
        "Justice:Sue": {
            "event subtype": "sue",
            "event type": "Justice:Sue",
            "event description": "The event is related to a court proceeding that has been initiated and someone sue the other.",
            "valid roles": ["Defendant", "Plaintiff", "Place", "Adjudicator"], 
        }, 
        "Justice:Trial-Hearing": {
            "event subtype": "trial hearing",
            "event type": "Justice:Trial-Hearing",
            "event description": "The event is related to a trial or hearing for someone.",
            "valid roles": ["Defendant", "Prosecutor", "Place", "Adjudicator"], 
        }, 
        "Life:Be-Born": {
            "event subtype": "born",
            "event type": "Life:Be-Born",
            "event description": "The event is related to life and someone is given birth to.",
            "valid roles": ["Person", "Place"], 
        }, 
        "Life:Die": {
            "event subtype": "die",
            "event type": "Life:Die",
            "event description": "The event is related to life and someone died.",
            "valid roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Life:Divorce": {
            "event subtype": "divorce",
            "event type": "Life:Divorce",
            "event description": "The event is related to life and someone was divorced.",
            "valid roles": ["Person", "Place"], 
        }, 
        "Life:Injure": {
            "event subtype": "injure",
            "event description": "The event is related to life and someone is injured.",
            "valid roles": ["Agent", "Victim", "Instrument", "Place"],
        },
        "Life:Marry": {
            "event subtype": "marry",
            "event type": "Life:Marry",
            "event description": "The event is related to life and someone is married.",
            "valid roles": ["Person", "Place"], 
        },
        "Manufacture:Artifact": {
            "event subtype": "artifact",
            "event type": "Manufacture:Artifact",
            "event description": "The event occurs whenever a person or an organization builds or manufactures a facility or a weapon, etc.",
            "valid roles": ["Artifact", "Agent", "Place"], 
        }, 
        "Movement:Transport-Artifact": {
            "event subtype": "transport artifact",
            "event type": "Movement:Transport-Artifact",
            "event description": "The event is related to movement. The event occurs when an artifact, like items or weapon, is moved from one place to another.",
            "valid roles": ["Artifact", "Destination", "Origin", "Agent", "Instrument"], 
        },
        "Movement:Transport-Person": {
            "event subtype": "transport person",
            "event type": "Movement:Transport-Person",
            "event description": "The event is related to movement. The event occurs when a person moves or is moved from one place to another.",
            "valid roles": ["Person", "Destination", "Origin", "Agent", "Instrument"], 
        },
        "Personnel:Elect": {
            "event subtype": "elect",
            "event type": "Personnel:Elect",
            "event description": "The event is related to a candidate wins an election.",
            "valid roles": ["Person", "Agent", "Place"], 
        },
        "Personnel:End-Position": {
            "event subtype": "end position",
            "event type": "Personnel:End-Position",
            "event description": "The event is related to a person stops working for an organization or a hiring manager.",
            "valid roles": ["Person", "Entity", "Place"],
        }, 
        "Personnel:Nominate": {
            "event subtype": "nominate",
            "event type": "Personnel:Nominate",
            "event description": "The event is related to a person being nominated for a position.",
            "valid roles": ["Person", "Agent"], 
        },  
        "Personnel:Start-Position": {
            "event subtype": "start position",
            "event type": "Personnel:Start-Position",
            "event description": "The event is related to a person begins working for an organization or a hiring manager.",
            "valid roles": ["Person", "Entity", "Place"], 
        },
        "Transaction:Transaction": {
            "event subtype": "transaction",
            "event type": "Transaction:Transaction",
            "event description": "The event is related to transaction. The event occurs when someone is giving, receiving, borrowing, or lending something when you cannot tell whether it is money or an asset in the context.",
            "valid roles": ["Giver", "Recipient", "Place", "Beneficiary"], 
        },
        "Transaction:Transfer-Money": {
            "event subtype": "transfer money",
            "event type": "Transaction:Transfer-Money",
            "event description": "The event is related to transaction. The event occurs when someone is giving, receiving, borrowing, or lending money.",
            "valid roles": ["Giver", "Recipient", "Place", "Beneficiary"], 
        },  
        "Transaction:Transfer-Ownership": {
            "event subtype": "transfer ownership",
            "event type": "Transaction:Transfer-Ownership",
            "event description": "The event is events refer to the buying, selling, loaning, borrowing, giving, receiving, bartering, stealing, or renting of physical items, assets,or organizations.",
            "valid roles": ["Giver", "Recipient", "Place", "Beneficiary", "Thing"], 
        }, 
    },
}