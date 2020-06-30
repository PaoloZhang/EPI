#!/usr/bin/env python3
from typing import List
import math

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution:
    
    def insert(self, nums: List[int], m: int, number: int, pos: int) -> int:
        if pos == m:
            nums[m] = number
            return m + 1

        index = m - 1
        while index >= pos:
            nums[index + 1] = nums[index]
            index -= 1
        nums[pos] = number
        return m+1

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """

        index = 0
        merged_len = m
        for i in range(n):
            element = nums2[i]
            k = index
            inserted = False
            while k < merged_len:
                if element < nums1[k]:
                    merged_len = self.insert(nums1, merged_len, element, k)

                    index = k+1
                    inserted = True
                    break
                k += 1
            if inserted == False:
                merged_len = self.insert(
                    nums1, merged_len, element, merged_len)
                index = merged_len - 1

    def replaceElements(self, arr: List[int]) -> List[int]:
        n = len(arr)
        max_num = arr[n-1]
        i = n - 2
        while i >= 0:
            if arr[i+1] > max_num:
                max_num = arr[i+1]
            erase_val = arr[i]
            arr[i] = max_num
            if erase_val > max_num:
                max_num = erase_val
            i -= 1
        arr[n-1] = -1
        return arr


    def reverse(self, x: int) -> int:
        y = 0
        is_negative = x < 0
        x = abs(x)
        while x > 0:
            y = y*10 + x % 10
            x = int(x/10)
        if is_negative:
            y = -y
        
        if y < -1*pow(2,31) or y > pow(2,31) - 1:
            y = 0
        return y
    def test(self,a:List[int]) -> None:
        b = enumerate(a)
        dict1 = {val:index for index,val in b }
        print(a ,'->',dict1)

    
    #Top->Bottom  Left->Right
    def connect(self, root: 'Node') -> 'Node':
        #It's a perfect tree, so left node and right node both exist.
        #node = root
        if root and root.left:
            root.left.next = root.right
        
        if root.next:
            root.right.next = root.next.left
        
        #self.connect(root.left)
        #self.connect(node.right)
            
        return root

    def longestPalindrome(self, s: str) -> str:
        def isSymmetric(s:str) -> bool:
            length = len(s)
            if length <= 1:
                return True
            
            half = int(length/2)
            
            return s[0:half+1] == s[-half:][::-1]
            '''
            for i in range(half):
                if s[i] != s[length - i -1]:
                    return False
            return True
            '''
        longest = 0
        longestSubString = ""

        if len(s) == 0:
            return longestSubString
        
        subString = ""
        for start_index in range(len(s)):
            for end_index in range(start_index,len(s)):
                subString = s[start_index:end_index + 1]
                
                if isSymmetric(subString) :
                    if longest < len(subString) :
                        longest = len(subString) 
                        longestSubString = subString[:]
            if len(s) - start_index <= longest:
                break
        return longestSubString
      

def add(a,b):
    running_sum,carry_in, mask,temp_a,temp_b =  0,0,1,a,b 
    while temp_a or temp_b:
        ak,bk = a&mask, b&mask
        carry_out = (ak & bk)|(ak & carry_in)|(bk & carry_in)
        running_sum |= ak^bk^carry_in
        carry_in,mask,temp_a,temp_b =(carry_out<<1,mask<<1,temp_a>>1,temp_b>>1)
    return running_sum | carry_in
        
def multiply(x:int,y:int) -> int:
    sum = 0
    while x != 0:
        if x & 1 == 1:
            sum = add(sum,y)
            x,y = x>>1,y<<1
    return sum 

def divide(x,y):
    result, power = 0, 32
    #为了除数足够大
    y_power = y<<power
    while x >= y:
        while y_power > x:
            y_power >> 1
            power -= 1
        
        result += 1<<power
        x -= y_power
    return result
        
'''
verison 0:
1. 第一遍先把小于pivot的放在前面
2. 第二遍再把大于pivot的放在后面
3. 等于pivot的自然在中间
'''
def dutch_flag_partition_v0(pivot_index:int, A:List[int]) -> None:
    pivot = A[pivot_index]
    length = len(A)

    for i in range(length):
        for j in range(i+1,length):
            if A[j] < pivot:
                A[i],A[j] = A[j],A[i]
                break
    print(A)

    for i in reversed(range(length)):
        if  A[i] < pivot:
            break
        for j in reversed(range(i)):
            if A[j] > pivot:
                A[i],A[j] = A[j],A[i]
                break
    print(A)

'''
用了双游标方案，所以复杂度少了一个数量级。
'''
def dutch_flag_partition_v1(pivot_index:int,A:List[int]) -> None:
    length = len(A)

    smaller = 0
    larger = length - 1
    pivot = A[pivot_index]

    for i in range(length):
        if A[i] < pivot:
            A[i],A[smaller] = A[smaller],A[i]
            smaller +=1
    for i in reversed(range(length)):
        if A[i] < pivot:
            break
        elif A[i] > pivot:
            A[i],A[larger] = A[larger],A[i]
            larger-=1
    print(A)


'''
equal是比较的浮标。
'''
def dutch_flag_partition_v2(pivot_index:int,A:List[int]) -> None:
    length = len(A)
    pivot = A[pivot_index]

    smaller,equal,larger = 0,0,length - 1
    while equal <= larger:
        if A[equal] < pivot:
            A[smaller],A[equal] = A[equal],A[smaller]
            smaller+=1
            equal+=1
        elif A[equal] == pivot:
            equal+=1
        elif A[equal] > pivot:
            A[equal],A[larger] = A[larger],A[equal]
            larger -= 1
            #Note! 此处浮标不应加1，具体参考EPI P.42
            #equal += 1
    print(A)

def quartile(A:List[int],unique:List[int]):
    length = len(A)
    a1_index,  a2_index = 0, length - 1
    while len(unique) >= 2:
        a1 = unique.pop(0)
        a2 = unique.pop()
        
        unkown_index = a1_index

        while unkown_index <= a2_index:
            if A[unkown_index] == a1:
                A[a1_index],A[unkown_index] = A[unkown_index],A[a1_index]
                unkown_index,a1_index = unkown_index+1,a1_index+1
            elif A[unkown_index] == a2:
                A[unkown_index],A[a2_index] = A[a2_index],A[unkown_index]
                a2_index -= 1
            else:
                unkown_index +=1
    print(A)

def add_binary(s1:List[int],s2:List[int])->None:
    len1,len2 = len(s1),len(s2)
    if len1 < len2:
        s1,s2 = s2,s1
        len1,len2 = len2,len1
    carry_in = 0

    j = 0
    for i in reversed(range(len1)):
        if j < len2:
            s1[i] = s1[i] + s2[len2-1-j]+carry_in
            carry_in = s1[i]//2   
            s1[i] = s1[i]%2
            j+=1
        else:
            s1[i] = s1[i] + carry_in
            carry_in = s1[i] // 2
            s1[i] = s1[i] % 2
            if carry_in == 0:
                break
    if carry_in > 0:
        s1.insert(0,carry_in)

def remove_leading_zeros(A:List[int]) -> List[int]:
    '''
    1.从左往右，获取第一个非零的元素index，如果全部为0，则index为list长度
    2.截取index之后的sublist
    3.获取的sublist若为空([])，则为[0]
    '''
    #A = A[next( (i for i,x in enumerate(A) if x!=0),len(A)):] or [0]
    A = A[next((i for i,x in enumerate(A) if x!=0),len(A)):] or [0]
    return(A)

def multiply(num1:List[int],num2:List[int]) -> List[int]:
    len1,len2 = len(num1),len(num2)
    sign = 1
    if num1[0] * num2[0] < 0:
        sign = -1
    
    result = [0] * (len1 + len2)

    num1[0],num2[0] = abs(num1[0]),abs(num2[0])

    for i in reversed(range(len1)):
        for j in reversed(range(len2)):
            result[i+j+1] += (num1[i]*num2[j])
            result[i+j] += result[i+j+1] // 10
            result[i+j+1] = result[i+j+1] % 10
    
    result = remove_leading_zeros(result)

    result[0] *= sign

    return result

  
def can_reach_end(A:List[int]) -> bool:
    furthest, i, last_index = 0, 0, len(A) - 1
    while furthest >=i and furthest < last_index:
        furthest = max(furthest, A[i] + i)
        i+=1
    return furthest >= last_index 

def jump(A:List[int]) -> int:
    length = len(A)
    if length < 2:
        return 0
    current_max, next_max, i,steps = A[0],0,0,0

    while current_max >= i and i < length:
        steps += 1
        while i <= current_max and i < length:
            next_max = max(next_max, i + A[i])
            i+=1
        current_max = next_max

    return steps

def remove_key(nums:List[int],key:int)->int:
    if not nums:
        return 0
    write_index = 0
    read_index = 0
    for read_index in range(len(nums)):
        if nums[read_index] != key:
            nums[write_index] = nums[read_index]
            write_index += 1
    return write_index

def get_single(A:List[int])->int:
    left,mid,right = 0,(len(A) - 1)//2+1,len(A)-1
    while right -left + 1> 3:
        mid = (right + left)//2 
        if A[mid] == A[mid+1]:
            mid += 1
        
        if (mid -left + 1)%2 == 1:
            right = mid
        else:
            left = mid +1 

    mid = (left + right)//2
    if A[left] == A[mid]:
        return right
    elif A[mid] == A[right]:
        return left
    else:
        return mid


def median(A, B):
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0
    
def get_max_profit_buy_sell_once(prices:List[int]) -> int:
    previous_low_price, max_profit = float('inf'),0
    for price in prices:
        max_profit = max(max_profit,price - previous_low_price)
        if price < previous_low_price:
            previous_low_price = price
    return max_profit
             

        
if __name__ == '__main__':
    A = [1,3] 
    B = [2]
    ans = median(A,B)
    print(ans)