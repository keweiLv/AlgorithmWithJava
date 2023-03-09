package prepareForByte;

/**
 * @author Kezi
 * @date 2023年03月09日 22:38
 * @describe 区域和检索 - 数组不可变
 */
public class NumArrayTwo {
	int[] tree;

	int lowbit(int x) {
		return x & -x;
	}

	int query(int x) {
		int ans = 0;
		for (int i = x; i > 0; i -= lowbit(i)) {
			ans += tree[i];
		}
		return ans;
	}

	void add(int x, int v) {
		for (int i = x; i <= n; i += lowbit(i)) {
			tree[i] += v;
		}
	}

	int[] nums;
	int n;

	public NumArrayTwo(int[] _nums) {
		nums = _nums;
		n = nums.length;
		tree = new int[n + 1];
		for (int i = 0; i < n; i++) {
			add(i + 1, nums[i]);
		}
	}

	public void update(int index, int val) {
		add(index + 1, val - nums[index]);
		nums[index] = val;
	}

	public int sumRange(int left, int right) {
		return query(right + 1) - query(left);
	}
}
