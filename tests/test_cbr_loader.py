import unittest

from src.cbr_loader import parse_history_records


class CBRLoaderTest(unittest.TestCase):
    def test_parser_keeps_nominal_and_normalizes_rate(self):
        xml = b"""<?xml version='1.0' encoding='windows-1251'?>
        <ValCurs ID='R01670'>
          <Record Date='01.02.2024' Id='R01670'>
            <Nominal>10</Nominal><Value>82,5000</Value>
          </Record>
        </ValCurs>"""

        records = parse_history_records(xml, "TJS")

        self.assertEqual(records.loc[0, "nominal"], 10)
        self.assertAlmostEqual(records.loc[0, "raw_rate"], 82.5)
        self.assertAlmostEqual(records.loc[0, "normalized_rate"], 8.25)


if __name__ == "__main__":
    unittest.main()
