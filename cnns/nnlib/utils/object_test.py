from cnns.nnlib.utils.object import Object

import unittest


class TestGetComplexMask(unittest.TestCase):

    def test_add(self):
        obj1 = Object()
        obj1.a = "1"

        obj2 = Object()
        obj2.c = "3"
        obj2.d = "4"

        obj1.add(obj2, prefix="new_")

        self.assertEqual(obj1.new_c, "3")
        self.assertEqual(obj1.new_d, "4")
        self.assertEqual(obj1.a, "1")

    def test_exception_attribute_already_exists(self):
        obj1 = Object()
        obj1.a = "1"

        obj2 = Object()
        obj2.a = "2"

        with self.assertRaises(Exception) as context:
            obj1.add(obj2)

        self.assertTrue(
            "The attribute: a already exists." in str(context.exception))


if __name__ == '__main__':
    unittest.main()
