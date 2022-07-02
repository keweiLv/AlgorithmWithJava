import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * @Author: Kezi
 * @Date: 2022/5/11 21:37
 */
public class Solution {

	// 序列化与反序列化二叉搜索树
	public String serialize(TreeNode root) {
		if (root == null) {
			return null;
		}
		List<String> list = new ArrayList<>();
		dfs1(root, list);
		int n = list.size();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++) {
			sb.append(list.get(i));
			if (i != n - 1) {
				sb.append(",");
			}
		}
		return sb.toString();
	}

	private void dfs1(TreeNode root, List<String> list) {
		if (root == null) {
			return;
		}
		list.add(String.valueOf(root.val));
		dfs1(root.left, list);
		dfs1(root.right, list);
	}

	public TreeNode deserialize(String s) {
		if (s == null) {
			return null;
		}
		String[] ss = s.split(",");
		return dfs2(0, ss.length - 1, ss);
	}

	private TreeNode dfs2(int l, int r, String[] ss) {
		if (l > r) {
			return null;
		}
		int ll = l + 1, rr = r, t = Integer.parseInt(ss[l]);
		while (ll < rr) {
			int mid = ll + rr >> 1;
			if (Integer.parseInt(ss[mid]) > t) {
				rr = mid;
			} else {
				ll = mid + 1;
			}
		}
		if (Integer.parseInt(ss[rr]) <= t) {
			rr++;
		}
		TreeNode ans = new TreeNode(t);
		ans.left = dfs2(l + 1, rr - 1, ss);
		ans.right = dfs2(rr, r, ss);
		return ans;
	}

	// 二查搜索树中第K小的元素
	public int kthSmallest(TreeNode root, int k) {
		Deque<TreeNode> stack = new ArrayDeque<>();
		while (root != null || !stack.isEmpty()) {
			while (root != null) {
				stack.addLast(root);
				root = root.left;
			}
			root = stack.pollLast();
			if (--k == 0) {
				return root.val;
			}
			root = root.right;
		}
		return -1;
	}

	// 单值二叉树
	int val = -1;

	public boolean isUnivalTree(TreeNode root) {
		if (val == -1) {
			val = root.val;
		}
		if (root == null) {
			return true;
		}
		if (root.val != val) {
			return false;
		}
		return isUnivalTree(root.left) && isUnivalTree(root.right);
	}


	//有序数组的平方
	public int[] sortedSquares(int[] nums) {
		int n = nums.length;
		int[] ans = new int[n];
		for (int i = 0, j = n - 1, pos = n - 1; i <= j; ) {
			if (nums[i] * nums[i] > nums[j] * nums[j]) {
				ans[pos] = nums[i] * nums[i];
				++i;
			} else {
				ans[pos] = nums[j] * nums[j];
				--j;
			}
			--pos;
		}
		return ans;
	}

	// 单词距离
	public int findClosest(String[] ws, String a, String b) {
		int n = ws.length, ans = n;
		for (int i = 0, p = -1, q = -1; i < n; i++) {
			String t = ws[i];
			if (t.equals(a)) {
				p = i;
			}
			if (t.equals(b)) {
				q = i;
			}
			if (p != -1 && q != -1) {
				ans = Math.min(ans, Math.abs(p - q));
			}
		}
		return ans;
	}

	// 两数之和Ⅱ
	public int[] twoSum(int[] numbers, int target) {
		int i = 0;
		int j = numbers.length - 1;
		while (i < j) {
			int sum = numbers[i] + numbers[j];
			if (sum < target) {
				i++;
			} else if (sum > target) {
				j--;
			} else {
				return new int[]{i + 1, j + 1};
			}
		}
		return new int[]{-1, -1};
	}

	// 粉刷房子
	public int minCost(int[][] cs) {
		int n = cs.length;
		int a = cs[0][0], b = cs[0][1], c = cs[0][2];
		for (int i = 1; i < n; i++) {
			int d = Math.min(b, c) + cs[i][0];
			int e = Math.min(a, c) + cs[i][1];
			int f = Math.min(a, b) + cs[i][2];
			a = d;
			b = e;
			c = f;
		}
		return Math.min(a, Math.min(b, c));
	}

	// 对称的二叉树
	public boolean isSymmetric(TreeNode root) {
		return root == null ? true : recur(root.left, root.right);
	}

	private boolean recur(TreeNode left, TreeNode right) {
		if (left == null && right == null) {
			return true;
		}
		if (left == null || right == null || left.val != right.val) {
			return false;
		}
		return recur(left.left, right.right) && recur(left.right, right.left);
	}

	// 斐波那契数列
	public int fib(int n) {
		int a = 0, b = 1, sum;
		for (int i = 0; i < n; i++) {
			sum = (a + b) % 1000000007;
			a = b;
			b = sum;
		}
		return a;
	}

	// 最长特殊序列Ⅱ
	public int findLUSlength(String[] strs) {
		int n = strs.length;
		int ans = -1;
		for (int i = 0; i < n; i++) {
			boolean check = true;
			for (int j = 0; j < n; j++) {
				if (i != j && isSubseq(strs[i], strs[j])) {
					check = false;
					break;
				}
			}
			if (check) {
				ans = Math.max(ans, strs[i].length());
			}
		}
		return ans;
	}

	public boolean isSubseq(String s, String t) {
		int ptS = 0, psT = 0;
		while (ptS < s.length() && psT < t.length()) {
			if (s.charAt(ptS) == t.charAt(psT)) {
				++ptS;
			}
			++psT;
		}
		return ptS == s.length();
	}

	/**
	 * 质数排列
	 * 解题点:质数的放置方案数为 a!a!，而非质数的放置方案数为 b!b!，根据「乘法原理」总的放置方案数为 a! \times b!a!×b!
	 */
	static int MOD = (int) 1e9 + 7;
	static int[] cnts = new int[110];

	static {
		List<Integer> list = new ArrayList<>();
		for (int i = 2; i <= 100; i++) {
			boolean ok = true;
			for (int j = 2; j * j <= i; j++) {
				if (i % j == 0) {
					ok = false;
				}
			}
			if (ok) {
				list.add(i);
			}
			cnts[i] = list.size();
		}
	}

	public int numPrimeArrangements(int n) {
		int a = cnts[n], b = n - a;
		long ans = 1;
		for (int i = b; i > 1; i--) {
			ans = ans * i % MOD;
		}
		for (int i = a; i > 1; i--) {
			ans = ans * i % MOD;
		}
		return (int) ans;
	}

	// 为运算表达式设计优先级--有点难，理解解法但没掌握
	char[] cs;
	public List<Integer> diffWaysToCompute(String s) {
		cs = s.toCharArray();
		return dfs(0, cs.length - 1);
	}
	List<Integer> dfs(int l, int r) {
		List<Integer> ans = new ArrayList<>();
		for (int i = l; i <= r; i++) {
			if (cs[i] >= '0' && cs[i] <= '9'){
				continue;
			}
			List<Integer> l1 = dfs(l,i-1),l2 = dfs(i+1,r);
			for (int a : l1){
				for (int b :l2){
					int cur = 0;
					if (cs[i] == '+'){
						cur = a + b;
					}else if (cs[i] == '-'){
						cur = a - b;
					}else {
						cur = a * b;
					}
					ans.add(cur);
				}
			}
		}
		if (ans.isEmpty()){
			int cur = 0;
			for (int i = l;i<=r;i++){
				cur = cur * 10 + (cs[i] - '0');
			}
			ans.add(cur);
		}
		return ans;
	}
}