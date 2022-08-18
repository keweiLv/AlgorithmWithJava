package LC_StudyPlan75_Lv2;

import java.util.List;

/**
 * @author Kezi
 * @date 2022年07月31日 22:56
 */
public class Node {
	public int val;
	public List<Node> children;

	public Node() {
	}

	public Node(int _val) {
		val = _val;
	}

	public Node(int _val, List<Node> _children) {
		val = _val;
		children = _children;
	}
}
