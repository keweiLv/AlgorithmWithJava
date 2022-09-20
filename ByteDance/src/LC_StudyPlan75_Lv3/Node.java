package LC_StudyPlan75_Lv3;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Kezi
 * @date 2022年09月20日 23:00
 */
public class Node {
	public int val;
	public List<Node> neighbors;
	public Node() {
		val = 0;
		neighbors = new ArrayList<Node>();
	}
	public Node(int _val) {
		val = _val;
		neighbors = new ArrayList<Node>();
	}
	public Node(int _val, ArrayList<Node> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}
}
