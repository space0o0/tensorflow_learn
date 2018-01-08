input = [3, 2, 4]
target = 6



def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    raise Exception("xxxx")
    input.index()
    i = 0
    for num in nums:
        j = i + 1
        for twoNum in nums[i, :]:
            if num + twoNum == target:
                return [i, j]
            j = j + 1
        i = i + 1


if __name__ == "__main__":
    twoSum(input, target)
