class Solution(object):
    s = "([]))"
    def isValid(self, s):
        stack = []
        for i in s:

            if i == "(" or i == "[" or i == "{":
                stack.append(i)
            elif i == ")" and stack[-1] == "(":
                stack.pop()
            elif i == "]" and stack[-1] == "[":
                stack.pop()
            elif i == "}" and stack[-1] == "{":
                stack.pop()
            else:
                return False

        if len(stack) != 0:
            return False
        return True

    print(isValid(object,s))





