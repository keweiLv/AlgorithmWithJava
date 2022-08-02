package daily;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年07月23日 22:39
 */
public class SolutionTwo {

	// 公交站台的距离
	public int distanceBetweenBusStops(int[] dist, int s, int t) {
		int n = dist.length, i = s, j = s, a = 0, b = 0;
		while (i != t) {
			a += dist[i];
			if (++i == n) {
				i = 0;
			}
		}
		while (j != t) {
			if (--j < 0) {
				j = n - 1;
			}
			b += dist[t];
		}
		return Math.min(a, b);
	}

	// 出现频率最高的K个数字
	public int[] topKFrequent(int[] nums, int k) {
		Map<Integer, Integer> map = new HashMap<>();
		for (int num : nums) {
			map.put(num, map.getOrDefault(num, 0) + 1);
		}
		// 使用优先队列
		PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			int num = entry.getKey(), cnt = entry.getValue();
			pq.offer(new int[]{num, cnt});
			if (pq.size() > k) {
				pq.poll();
			}
		}
		int[] ans = new int[pq.size()];
		for (int i = 0; i < k; i++) {
			ans[i] = pq.poll()[0];
		}
		return ans;
	}

	// 山峰数组额顶部
	public int peakIndexInMountainArray(int[] arr) {
		int n = arr.length;
		int left = 1, right = n - 2, ans = 0;
		while (left <= right) {
			int mid = (left + right) / 2;
			if (arr[mid] > arr[mid + 1]) {
				ans = mid;
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		return ans;
	}

	// 排序数组中只出现一次的数字
	public int singleNonDuplicate(int[] nums) {
		int n = nums.length, l = 0, r = n - 1;
		int ans = -1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (mid < n - 1 && nums[mid] == nums[mid + 1]) {
				if (mid % 2 == 0) {
					l = mid + 2;
				} else {
					r = mid - 1;
				}
			} else if (mid > 0 && nums[mid] == nums[mid - 1]) {
				if (mid % 2 == 0) {
					r = mid - 2;
				} else {
					l = mid + 1;
				}
			} else {
				ans = nums[mid];
				break;
			}
		}
		return ans;
	}

	/**
	 * 有效的正方形
	 * 该图形是正方形，那么任意三点组成的一定是等腰直角三角形，用此条件作为判断
	 */
	long len = -1;

	public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
		return calc(p1, p2, p3) && calc(p1, p2, p4) && calc(p1, p3, p4) && calc(p2, p3, p4);
	}

	boolean calc(int[] a, int[] b, int[] c) {
		long l1 = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
		long l2 = (a[0] - c[0]) * (a[0] - c[0]) + (a[1] - c[1]) * (a[1] - c[1]);
		long l3 = (b[0] - c[0]) * (b[0] - c[0]) + (b[1] - c[1]) * (b[1] - c[1]);
		boolean ok = (l1 == l2 && l1 + l2 == l3) || (l1 == l3 && l1 + l3 == l2) || (l2 == l3 && l2 + l3 == l1);
		if (!ok) {
			return false;
		}
		if (len == -1) {
			len = Math.min(l1, l2);
		} else if (len == 0 || len != Math.min(l1, l2)) {
			return false;
		}
		return true;
	}

	// 链表的中间节点
	public ListNode middleNode(ListNode head) {
		ListNode slow = head, fast = head;
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		return slow;
	}

	// 环形链表Ⅱ
	public ListNode detectCycle(ListNode head) {
		ListNode fast = head, slow = head;
		while (true) {
			if (fast == null || fast.next == null) {
				return null;
			}
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				break;
			}
		}
		fast = head;
		while (slow != fast) {
			slow = slow.next;
			fast = fast.next;
		}
		return fast;
	}

	// 层内元素之和
	public int maxLevelSum(TreeNode root) {
		int ans = 1, maxSum = root.val;
		List<TreeNode> q = new ArrayList<>();
		q.add(root);
		for (int lev = 1; !q.isEmpty(); ++lev) {
			List<TreeNode> nq = new ArrayList<>();
			int sum = 0;
			for (TreeNode node : q) {
				sum += node.val;
				if (node.left != null) {
					nq.add(node.left);
				}
				if (node.right != null) {
					nq.add(node.right);
				}
			}
			if (sum > maxSum) {
				maxSum = sum;
				ans = lev;
			}
			q = nq;
		}
		return ans;
	}

	// 生成每种字符串都是奇数个的字符串
	public String generateTheString(int n) {
		StringBuilder sb = new StringBuilder();
		if (n % 2 == 0 && --n >= 0) {
			sb.append("a");
		}
		while (n-- > 0) {
			sb.append("b");
		}
		return sb.toString();
	}

	//

}