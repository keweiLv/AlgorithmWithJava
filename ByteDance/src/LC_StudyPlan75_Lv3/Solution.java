package LC_StudyPlan75_Lv3;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Kezi
 * @date 2022年09月08日 0:02
 */
public class Solution {

	/**
	 * 位1的个数
	 * 算术右移 >> ：舍弃最低位，高位用符号位填补；
	 * 逻辑右移 >>> ：舍弃最低位，高位用 0 填补
	 * 对于负数而言，其二进制最高位是 1，如果使用算术右移，那么高位填补的仍然是 1。也就是 n 永远不会为 0
	 */
	public int hammingWeight(int n) {
		int count = 0;
		while (n != 0) {
			count += n & 1;
			n >>>= 1;
		}
		return count;
	}

	// 只出现一次的数字
	public int singleNumber(int[] nums) {
		int ans = nums[0];
		for (int i = 1; i < nums.length; i++) {
			ans = ans ^ nums[i];
		}
		return ans;
	}

	// 子集Ⅱ
	private List<List<Integer>> ans;
	private List<Integer> path;

	public List<List<Integer>> subsetsWithDup(int[] nums) {
		ans = new ArrayList<>();
		path = new ArrayList<>();
		Arrays.sort(nums);
		int n = nums.length;
		boolean[] visited = new boolean[n];
		// 开始回溯
		backtrace(nums, 0, visited, n);
		return ans;
	}

	private void backtrace(int[] nums, int start, boolean[] visited, int n) {
		ans.add(new ArrayList<>(path));
		for (int i = start; i < n; i++) {
			if (i > 0 && nums[i - 1] == nums[i] && !visited[i - 1]) {
				continue;
			}
			visited[i] = true;
			path.add(nums[i]);
			backtrace(nums, i + 1, visited, n);
			visited[i] = false;
			path.remove(path.size() - 1);
		}
	}

	// 字母异位词分组
	public List<List<String>> groupAnagrams(String[] strs) {
		return new ArrayList<>(Arrays.stream(strs)
				.collect(Collectors.groupingBy(str -> {
					char[] chars = str.toCharArray();
					Arrays.sort(chars);
					return new String(chars);
				})).values()
		);
	}

	// 实现strStr（）
	public int strStr(String ss, String pp) {
		int n = ss.length(), m = pp.length();
		char[] s = ss.toCharArray(), p = pp.toCharArray();
		for (int i = 0; i <= n - m; i++) {
			int a = i, b = 0;
			while (b < m && s[a] == p[b]) {
				a++;
				b++;
			}
			if (b == m) {
				return i;
			}
		}
		return -1;
	}

	// 账户合并
	public List<List<String>> accountsMerge(List<List<String>> accounts) {
		Map<String, Integer> emailToId = new HashMap<>();
		int n = accounts.size();
		UnionFind myUnion = new UnionFind(n);
		for (int i = 0; i < n; i++) {
			int num = accounts.get(i).size();
			for (int j = 1; j < num; j++) {
				String curEmail = accounts.get(i).get(j);
				if (!emailToId.containsKey(curEmail)) {
					emailToId.put(curEmail, i);
				} else {
					myUnion.union(i, emailToId.get(curEmail));
				}
			}
		}
		Map<Integer, List<String>> idToEmail = new HashMap<>();
		for (Map.Entry<String, Integer> entry : emailToId.entrySet()) {
			int id = myUnion.find(entry.getValue());
			List<String> emails = idToEmail.getOrDefault(id, new ArrayList<>());
			emails.add(entry.getKey());
			idToEmail.put(id, emails);
		}
		List<List<String>> res = new ArrayList<>();
		for (Map.Entry<Integer, List<String>> entry : idToEmail.entrySet()) {
			List<String> emails = entry.getValue();
			Collections.sort(emails);
			List<String> tmp = new ArrayList<>();
			tmp.add(accounts.get(entry.getKey()).get(0));
			tmp.addAll(emails);
			res.add(tmp);
		}
		return res;
	}

	class UnionFind {
		int[] parent;

		public UnionFind(int n) {
			parent = new int[n];
			for (int i = 0; i < n; i++) {
				parent[i] = i;
			}
		}

		public void union(int index1, int index2) {
			parent[find(index2)] = find(index1);
		}

		public int find(int index) {
			if (parent[index] != index) {
				parent[index] = find(parent[index]);
			}
			return parent[index];
		}
	}

	// 两两交换链表中的节点
	public ListNode swapPairs(ListNode head) {
		if (head == null || head.next == null) {
			return head;
		}
		ListNode next = head.next;
		head.next = swapPairs(next.next);
		next.next = head;
		return next;
	}

	// 特殊数组的特征值
	public int specialArray(int[] nums) {
		Arrays.sort(nums);
		if (nums[0] >= nums.length) {
			return nums.length;
		}
		for (int i = 1, res = 0; i < nums.length; i++) {
			if (nums[i] >= (res = nums.length - i) && nums[i - 1] < res) {
				return res;
			}
		}
		return -1;
	}

	// 罗马数字转整数
	/*public int romanToInt(String s) {
		int sum = 0;
		int preNum = getValue(s.charAt(0));
		for (int i = 1; i < s.length(); i++) {
			int num = getValue(s.charAt(i));
			if (preNum < num) {
				sum -= preNum;
			} else {
				sum += preNum;
			}
			preNum = num;
		}
		sum += preNum;
		return sum;
	}

	private int getValue(char charAt) {
		return switch (charAt) {
			case 'I' -> 1;
			case 'V' -> 5;
			case 'X' -> 10;
			case 'L' -> 50;
			case 'C' -> 100;
			case 'D' -> 500;
			case 'M' -> 1000;
			default -> 0;
		};
	}*/

	// 克隆图
	private HashMap<Node, Node> visited = new HashMap<>();

	public Node cloneGraph(Node node) {
		if (node == null) {
			return node;
		}
		if (visited.containsKey(node)) {
			return visited.get(node);
		}
		Node clone = new Node(node.val, new ArrayList<>());
		visited.put(node, clone);
		for (Node neighbor : node.neighbors) {
			clone.neighbors.add(cloneGraph(neighbor));
		}
		return clone;
	}

	// 每日温度-单调栈
	public int[] dailyTemperatures(int[] T) {
		Deque<Integer> stack = new LinkedList<>();
		int length = T.length;
		int[] ans = new int[length];
		for (int i = 0; i < length; i++) {
			int temp = T[i];
			while (!stack.isEmpty() && temp > T[stack.peek()]) {
				int pre = stack.pop();
				ans[pre] = i - pre;
			}
			stack.push(i);
		}
		return ans;
	}

	/**
	 * 逆波兰表达式求值
	 * 如果遇到操作数，则将操作数入栈；
	 * 如果遇到运算符，则将两个操作数出栈，其中先出栈的是右操作数，后出栈的是左操作数，使用运算符对两个操作数进行运算，将运算得到的新操作数入栈。
	 */
	public int evalRPN(String[] tokens) {
		Deque<Integer> stack = new LinkedList<>();
		int n = tokens.length;
		for (int i = 0; i < n; i++) {
			String token = tokens[i];
			if (isNumber(token)) {
				stack.push(Integer.parseInt(token));
			} else {
				int num2 = stack.pop();
				int num1 = stack.pop();
				switch (token) {
					case "+":
						stack.push(num1 + num2);
						break;
					case "-":
						stack.push(num1 - num2);
						break;
					case "*":
						stack.push(num1 * num2);
						break;
					case "/":
						stack.push(num1 / num2);
						break;
					default:
				}
			}
		}
		return stack.pop();
	}

	private boolean isNumber(String token) {
		return !("+".equals(token) || "-".equals(token) || "*".equals(token) || "/".equals(token));
	}

	// 单词拆分
	public boolean wordBreak(String s, List<String> wordDict) {
		int len = s.length(), maxw = 0;
		boolean[] dp = new boolean[len + 1];
		dp[0] = true;
		Set<String> set = new HashSet<>();
		for (String str : wordDict) {
			set.add(str);
			maxw = Math.max(maxw, str.length());
		}
		for (int i = 1; i < len + 1; i++) {
			for (int j = i; j >= 0 && j >= i - maxw; j--) {
				if (dp[j] && set.contains(s.substring(j, i))) {
					dp[i] = true;
					break;
				}
			}
		}
		return dp[len];
	}

	// 子集
	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		backtrace(0, nums, res, new ArrayList<>());
		return res;
	}

	private void backtrace(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
		res.add(new ArrayList<>(tmp));
		for (int j = i; j < nums.length; j++) {
			tmp.add(nums[j]);
			backtrace(j + 1, nums, res, tmp);
			tmp.remove(tmp.size() - 1);
		}
	}

	// 寻找旋转排序数组中的最小值
	public int findMin(int[] nums) {
		int left = 0;
		int right = nums.length - 1;
		while (left < right) {
			int mid = left + (right - left) / 2;
			if (nums[mid] > nums[right]) {
				left = mid + 1;
			} else if (nums[mid] < nums[right]) {
				right = mid;
			}
		}
		return nums[left];
	}

	// 零矩阵
	public void setZeroes(int[][] mat) {
		int n = mat.length, m = mat[0].length;
		boolean[] rows = new boolean[n], cols = new boolean[m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (mat[i][j] == 0) {
					rows[i] = cols[j] = true;
				}
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (rows[i] || cols[j]){
					mat[i][j] = 0;
				}
			}
		}
	}
}

