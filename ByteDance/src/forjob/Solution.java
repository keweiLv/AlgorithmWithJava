package forjob;

/**
 * @author Kezi
 * @date 2022年10月10日 23:28
 */
public class Solution {
	//上下翻转二叉树
	public TreeNode upsideDownBinaryTree(TreeNode root) {
		TreeNode parent = null,parent_right = null;
		while (root != null){
			TreeNode root_left = root.left;
			TreeNode root_right = root.right;
			root.left = parent_right;
			root.right = parent;
			parent = root;
			root = root_left;
			parent_right = root_right;
		}
		return parent;
	}

	//  仅执行一次字符串交换能否使两个字符串相等
	public boolean areAlmostEqual(String s1, String s2) {
		int n = s1.length(),a=-1,b=-1;
		for (int i = 0;i<n;i++){
			if (s1.charAt(i) == s2.charAt(i)){
				continue;
			}
			if (a == -1){
				a = i;
			}else if (b == -1){
				b = i;
			}else {
				return false;
			}
		}
		if (a == -1){
			return true;
		}
		if (b == -1){
			return false;
		}
		return s1.charAt(a) == s2.charAt(b) && s1.charAt(b) == s2.charAt(a);
	}
}
