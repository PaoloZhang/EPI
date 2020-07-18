#!/usr/bin/env python3
from typing import List
import itertools
import math

class ListNode:
  def __init__(self, val=0, next=None):
    self.val = val
    self.next = next
  
def List2ListNode(nums:List[int])->ListNode:
  if len(nums) == 0:
    return None
  ans = ListNode(val=nums[0])

  node = ans
  for i in range(1,len(nums)):
    node.next = ListNode(nums[i])
    node = node.next
  return ans

def ListNode2List(l:ListNode) -> List[int]:
  ans = []
  while not l:
    ans.append(l.val)
    l = l.next
  return ans

    


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        def partition(intervals:List[List[int]], left:int, right:int) -> int:
            pivotkey = intervals[left]

            while left < right:
                while left < right and intervals[right][0] >= pivotkey[0]:
                    right -= 1
                intervals[left] = intervals[right]
                
                while left < right and intervals[left][0] <= pivotkey[0]:
                     left += 1
                intervals[right] = intervals[left]

                intervals[left] = pivotkey
            return left
        
        def qsort(intervals:List[List[int]], left:int,right:int) :
            if left < right:
                pivot_pos = partition(intervals,left,right)
                qsort(intervals,left, pivot_pos-1)
                qsort(intervals,pivot_pos+1,right)
            
            
        
        
        def sort(intervals:List[List[int]]) -> List[List[int]]:
            pivot = intervals[0]
            less = [x for x in intervals if x[0] <= pivot[0]]
            more = [y for y in intervals if y[0] > pivot[0]]
            return sort(less) + [pivot] + sort(more)
        
        def overlap(a:List[int],b:List[int]) -> bool:
            return not (a[1] < b[0] or b[1] < a[0])
        
        def merge(a:List[int],b:List[int]) ->List:
            return [min(a[0],b[0]),max(a[1],b[1])]
        
        
        length = len(intervals)
        ans = []
        if length == 0:
            return ans
        
        qsort(intervals,0,len(intervals)-1)
        ans.append(intervals[0])
        for read_index in range(1,len(intervals)):
            if   overlap(ans[len(ans) -1],intervals[read_index]):
                    ans[len(ans) - 1] = merge(ans[len(ans) -1],intervals[read_index])
            else:
                ans.append(intervals[read_index])
        
        return ans

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        
        if l1.val > l2.val:
            l2,l1 = l1,l2
        
        ans = l1
        
        while l1 and l1.next and l2:
            
          
            if l1.val <= l2.val and l1.next.val >= l2.val:
                tmp = l2.next
                l2.next = l1.next
                l1.next = l2
                l1 = l2
                l2 = tmp
          #if l1.val < l2.val and l1.next.val < l2.val:
            else:
                l1 = l1.next  
        
        if not l1.next:
            l1.next = l2
            
        return ans


    def isValid(self, s: str) -> bool:
        mapping = {'(':')','{':'}','[':']'}
        buffer = []
        for i in range(len(s)):
          if s[i] in mapping:
              buffer.append(s[i])
          else:
              if len(buffer) == 0:
                return False
              elif mapping.get(buffer.pop()) != s[i]:
                return False
        
        
        return len(buffer)==0

    def getMaxArray(self,a:List[int]) -> int:
      min_sum = max_sum = 0
      for running_sum in itertools.accumulate(a):
        min_sum = min(min_sum,running_sum)
        max_sum = max(max_sum,running_sum - min_sum)
      return max_sum


            
if __name__ =='__main__':
  a = [-200,106,-100,5,904,40,523,12,-335,-385,-124,481,-31]
  Solution().getMaxArray(a)
  