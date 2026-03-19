from __future__ import annotations

import random

from benchpress.models import ExpectedToolCall, PromptCase

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_inventory",
            "description": "Look up current inventory count for a product by SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Product SKU code"},
                    "warehouse": {
                        "type": "string",
                        "enum": ["east", "west", "central"],
                        "description": "Warehouse location",
                    },
                },
                "required": ["sku", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": "Calculate shipping cost between two zip codes for a given weight in pounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_zip": {
                        "type": "string",
                        "description": "Origin zip code",
                    },
                    "dest_zip": {
                        "type": "string",
                        "description": "Destination zip code",
                    },
                    "weight_lbs": {
                        "type": "number",
                        "description": "Package weight in pounds",
                    },
                },
                "required": ["origin_zip", "dest_zip", "weight_lbs"],
            },
        },
    },
]

SKU_POOL = [
    "WDG-4420",
    "ABC-1234",
    "XYZ-9900",
    "PLT-0071",
    "MNO-5588",
    "QRS-3321",
    "DEF-7766",
    "GHI-2299",
    "JKL-8844",
    "TUV-1155",
]

ZIP_POOL = [
    "90210",
    "10001",
    "60601",
    "30301",
    "98101",
    "02101",
    "33101",
    "85001",
    "73301",
    "44101",
]

WAREHOUSES = ["east", "west", "central"]

CONVERSATIONAL_PROMPTS = [
    "Explain the difference between FIFO and LIFO inventory methods in 2-3 sentences.",
    "What are the main factors that affect shipping costs for e-commerce businesses?",
    "Describe three best practices for warehouse organization.",
    "What is the difference between a SKU and a UPC code?",
    "Briefly explain what safety stock is and why it matters.",
    "What are the advantages of zone-based shipping pricing?",
    "Explain the concept of just-in-time inventory management.",
    "What is a bill of lading and when is it used?",
    "Describe the difference between cross-docking and traditional warehousing.",
    "What factors should a business consider when choosing between air and ground shipping?",
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a warehouse operations assistant. You help with inventory lookups "
        "and shipping calculations. Use the provided tools when the user asks about "
        "specific inventory or shipping information. For general knowledge questions, "
        "respond directly without using tools."
    ),
}


class PromptGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, n: int) -> list[PromptCase]:
        cases: list[PromptCase] = []
        for i in range(n):
            roll = self.rng.random()
            if roll < 0.4:
                cases.append(self._direct_tool(i))
            elif roll < 0.7:
                cases.append(self._reasoning_tool(i))
            else:
                cases.append(self._conversational(i))
        return cases

    def _direct_tool(self, idx: int) -> PromptCase:
        if self.rng.random() < 0.5:
            sku = self.rng.choice(SKU_POOL)
            warehouse = self.rng.choice(WAREHOUSES)
            prompt = f"Check the inventory for SKU {sku} at the {warehouse} warehouse."
            expected = [
                ExpectedToolCall(
                    function_name="lookup_inventory",
                    arguments={"sku": sku, "warehouse": warehouse},
                )
            ]
        else:
            origin = self.rng.choice(ZIP_POOL)
            dest = self.rng.choice([z for z in ZIP_POOL if z != origin])
            weight = round(self.rng.uniform(1, 50), 1)
            prompt = (
                f"Calculate shipping cost from {origin} to {dest} "
                f"for a {weight}-pound package."
            )
            expected = [
                ExpectedToolCall(
                    function_name="calculate_shipping",
                    arguments={
                        "origin_zip": origin,
                        "dest_zip": dest,
                        "weight_lbs": weight,
                    },
                )
            ]

        return PromptCase(
            id=f"prompt-{idx:03d}",
            category="direct_tool",
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": prompt}],
            tools=TOOLS,
            expected_tool_calls=expected,
        )

    def _reasoning_tool(self, idx: int) -> PromptCase:
        sku = self.rng.choice(SKU_POOL)
        warehouse = self.rng.choice(WAREHOUSES)
        origin = self.rng.choice(ZIP_POOL)
        dest = self.rng.choice([z for z in ZIP_POOL if z != origin])
        weight = round(self.rng.uniform(1, 50), 1)

        templates = [
            (
                f"I need to ship SKU {sku} from our {warehouse} warehouse to "
                f"zip code {dest}. The package weighs {weight} lbs. First check "
                f"if it's in stock, then calculate the shipping cost. Our "
                f"{warehouse} warehouse zip is {origin}.",
                [
                    ExpectedToolCall(
                        function_name="lookup_inventory",
                        arguments={"sku": sku, "warehouse": warehouse},
                    ),
                    ExpectedToolCall(
                        function_name="calculate_shipping",
                        arguments={
                            "origin_zip": origin,
                            "dest_zip": dest,
                            "weight_lbs": weight,
                        },
                    ),
                ],
            ),
            (
                f"A customer wants to know the shipping cost for a {weight}-pound "
                f"order going from {origin} to {dest}. They also want to confirm "
                f"that SKU {sku} is available at the {warehouse} warehouse.",
                [
                    ExpectedToolCall(
                        function_name="calculate_shipping",
                        arguments={
                            "origin_zip": origin,
                            "dest_zip": dest,
                            "weight_lbs": weight,
                        },
                    ),
                    ExpectedToolCall(
                        function_name="lookup_inventory",
                        arguments={"sku": sku, "warehouse": warehouse},
                    ),
                ],
            ),
            (
                f"Can you check if we have {sku} at the {warehouse} location? "
                f"If so, what would it cost to ship {weight} lbs from {origin} "
                f"to {dest}?",
                [
                    ExpectedToolCall(
                        function_name="lookup_inventory",
                        arguments={"sku": sku, "warehouse": warehouse},
                    ),
                    ExpectedToolCall(
                        function_name="calculate_shipping",
                        arguments={
                            "origin_zip": origin,
                            "dest_zip": dest,
                            "weight_lbs": weight,
                        },
                    ),
                ],
            ),
        ]

        prompt, expected = self.rng.choice(templates)

        return PromptCase(
            id=f"prompt-{idx:03d}",
            category="reasoning_tool",
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": prompt}],
            tools=TOOLS,
            expected_tool_calls=expected,
        )

    def _conversational(self, idx: int) -> PromptCase:
        prompt = self.rng.choice(CONVERSATIONAL_PROMPTS)
        return PromptCase(
            id=f"prompt-{idx:03d}",
            category="conversational",
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": prompt}],
            tools=TOOLS,
            expected_no_tool=True,
        )
