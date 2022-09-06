package LC_StudyPlan75_Lv2;

/**
 * @author Kezi
 * @date 2022年08月18日 22:13
 */
public class Solution {

	/**
	 * 快乐数
	 * 如果 n 是一个快乐数，即没有循环，那么快跑者最终会比慢跑者先到达数字 1。
	 * 如果 n 不是一个快乐的数字，那么最终快跑者和慢跑者将在同一个数字上相遇。
	 */
	public boolean isHappy(int n) {
		int slow = n, fast = squareSum(n);
		while (slow != fast) {
			slow = squareSum(slow);
			fast = squareSum(squareSum(fast));
		}
		return slow == 1;
	}

	private int squareSum(int n) {
		int sum = 0;
		while (n > 0) {
			int digit = n % 10;
			sum += digit * digit;
			n /= 10;
		}
		return sum;
	}

	// 最长公共前缀
	public String longestCommonPrefix(String[] strs) {
		if (strs.length == 0) {
			return "";
		}
		String ans = strs[0];
		for (int i = 1; i < strs.length; i++) {
			int j = 0;
			for (; j < ans.length() && j < strs[i].length(); j++) {
				if (ans.charAt(j) != strs[i].charAt(j)) {
					break;
				}
			}
			ans = ans.substring(0, j);
			if (ans.equals("")) {
				return ans;
			}
		}
		return ans;
	}

	// 回文链表
	public boolean isPalindrome(ListNode head) {
		if (head == null || head.next == null) {
			return true;
		}
		ListNode slow = head, fast = head;
		ListNode pre = head, prePre = null;
		while (fast != null && fast.next != null) {
			pre = slow;
			slow = slow.next;
			fast = fast.next.next;
			pre.next = prePre;
			prePre = pre;
		}
		if (fast != null) {
			slow = slow.next;
		}
		while (pre != null && slow != null) {
			if (pre.val != slow.val) {
				return false;
			}
			pre = pre.next;
			slow = slow.next;
		}
		return true;
	}

	// 奇偶链表
	public ListNode oddEvenList(ListNode head) {
		if (head == null) {
			return head;
		}
		ListNode evenHead = head.next;
		ListNode odd = head, even = evenHead;
		while (even != null && even.next != null) {
			odd.next = even.next;
			odd = odd.next;
			even.next = odd.next;
			even = even.next;
		}
		odd.next = evenHead;
		return head;
	}

	// 打家劫舍
	public int rob(int[] nums) {
		int pre = 0;
		int cur = 0;
		for (int num : nums) {
			int tmp = Math.max(cur, pre + num);
			pre = cur;
			cur = tmp;
		}
		return cur;
	}

	// 排序链表
	public ListNode sortList(ListNode head) {
		if (head == null || head.next == null) {
			return head;
		}
		ListNode fast = head.next, slow = head;
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		ListNode tmp = slow.next;
		slow.next = null;
		ListNode left = sortList(head);
		ListNode right = sortList(tmp);
		ListNode h = new ListNode(0);
		ListNode res = h;
		while (left != null && right != null) {
			if (left.val < right.val) {
				h.next = left;
				left = left.next;
			} else {
				h.next = right;
				right = right.next;
			}
			h = h.next;
		}
		h.next = left != null ? left : right;
		return res.next;
	}
}
