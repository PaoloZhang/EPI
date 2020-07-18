#!/usr/bin/env python3
from typing import List
import math
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        item = [None for i in range(n)]
        ans = []
        for i in range(n):
            ans.append(item.copy())
        
        i,j,total,SHIFT_NUM,count = 0,0,n**2,0,0
        
        SHIFT = [[0,1],[1,0],[0,-1],[-1,0]]
        
        while count < total:
            ans[i][j] = count + 1
            count += 1
            
            if count == total:
                break
            
            maybe_i,maybe_j = i + SHIFT[SHIFT_NUM % 4][0],j + SHIFT[SHIFT_NUM % 4][1]
            
            if maybe_i >= n or maybe_j>=n or ans[maybe_i][maybe_j] != None:
                SHIFT_NUM += 1
                maybe_i,maybe_j = i + SHIFT[SHIFT_NUM % 4][0],j + SHIFT[SHIFT_NUM % 4][1]
            
            i,j = maybe_i,maybe_j
        
        return ans
    def romanToInt(self, s: str) -> int:
        mapping = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        
        ans,prev,positive = 0,0,True

        
        for i in reversed(range(len(s))):
            val = mapping.get(s[i])
            if val > prev:
                positive = True
            elif val < prev:
                positive = False
            
            if positive:
                ans += val
            else:
                ans -= val
        
            prev = val
        
        return ans

    def longestPalindrome(self, s: str) -> str:
        def isSymmetric(s:str) -> bool:
            length = len(s)
            half = length//2
            return s[:half] == s[-half:][::-1]
            '''
            for i in range(half):
                if s[i] != s[length - i -1]:
                    return False
            return True
            '''
        longest = 0
        longestSubString = ""

        length = len(s)
        if length == 0:
            return longestSubString
        subString = ""
        for start_index in range(length):
            for end_index in range(start_index,length):
                subString = s[start_index:end_index+1]
                
                if isSymmetric(subString) :
                    if longest < len(subString) :
                        longest = len(subString) 
                        longestSubString = subString
            if length - start_index <= longest:
                break
        return longestSubString
            
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1,nums2 = nums2,nums1
        
        m,n = len(nums1),len(nums2)
        
        if m == 0:
            if n%2==1:
                return nums2[n//2]
            else:
                return (nums2[n//2] + nums2[n//2 -1])/2
            
        
        imin,imax,half_len = 0,m,(m+n+1)//2
        
        while imin<=imax:
            i = (imin + imax)//2
            j = half_len - i
            max_left = nums2[0]
            min_right = nums1[0]
            
            if i > 0 and nums1[i-1]>nums2[j]:
                imax -= 1
            elif i < m and nums1[i] < nums2[j - 1]:
                imin += 1
            
            else:
                if  i == 0:
                    max_left = nums2[j-1]
                    if j <= n -1:
                        min_right = min(nums2[j],nums1[0])
                    else:
                        min_right = nums1[0]
                elif i == m:
                    if j > 0:
                        max_left = max(nums1[i-1],nums2[j-1])
                    else:
                        max_left = nums1[i-1]
                        
                    min_right = nums2[j]
                else:
                    max_left = max(nums1[i-1],nums2[j-1])
                    min_right = min(nums1[i],nums2[j])
            
                if (m+n)%2==1:
                    return max_left
                else:
                    return (max_left + min_right)/2

    def find_longest_unique_str(self,s:str) -> str:
        begin,end,longest = 0,0,1
        reserved_begin, reserved_end = begin,end
        while len(s) - begin > longest:
          end += 1
          for j in range(begin,end):
            if s[j] == s[end]:
              begin = j + 1
              break
          if end - begin + 1 > longest:
            longest = end - begin + 1
            reserved_begin,reserved_end = begin,end
        
        return s[reserved_begin:reserved_end+1]
      
if __name__ =='__main__':
   # ans = Solution().find_longest_unique_str('abcabcbbabcd')
   # print(ans)
   for  i in range(1,2):
     print(i)