package planToByte;

/**
 * @author Kezi
 * @date 2023年02月03日 23:07
 */
public class NewSolution {

	// 二叉树着色游戏
	private int x, lsz, rsz;
	public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
		this.x = x;
		dfs(root);
		return Math.max(Math.max(lsz, rsz), n - 1 - lsz - rsz) * 2 > n;
	}
	private int dfs(TreeNode node) {
		if (node == null) {
			return 0;
		}
		int ls = dfs(node.left);
		int rs = dfs(node.right);
		if (node.val == x) {
			lsz = ls;
			rsz = rs;
		}
		return ls + rs + 1;
	}
}
