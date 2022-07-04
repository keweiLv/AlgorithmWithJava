package raceCollection;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Kezi
 * @date 2022年07月03日 10:47
 * 第三百场周赛-蔚来
 */
public class The300 {

	public static void main(String[] args) {

	}

	/**
	 * 解密消息
	 *
	 * @param key
	 * @param message
	 * @return
	 */
	public static String decodeMessage(String key, String message) {
		Map<Character, Integer> table = new HashMap<>(16);
		int cunt = 0;
		for (int i = 0; i < key.length(); i++) {
			if (table.containsKey(key.charAt(i)) || key.charAt(i) == ' ') {
				continue;
			}
			table.put(key.charAt(i), cunt++);
		}
		char[] chars = message.toCharArray();
		StringBuilder ans = new StringBuilder();
		for (char aChar : chars) {
			if (aChar == ' ') {
				ans.append(' ');
				continue;
			}
			char c = (char) (table.get(aChar) + 'a');
			ans.append(c);
		}
		return ans.toString();
	}

	public class ListNode {
		int val;
		ListNode next;

		ListNode() {
		}

		ListNode(int val) {
			this.val = val;
		}

		ListNode(int val, ListNode next) {
			this.val = val;
			this.next = next;
		}
	}

	/**
	 * 螺旋矩阵 IV
	 */
	public int[][] spiralMatrix(int m, int n, ListNode head) {
		int[][] matrix = new int[m][n];
		for (int i = 0; i < m; i++) {
			Arrays.fill(matrix[i], -1);
		}
		int x = 0, y = 0;
		while (head.next != null) {
			while (head.next != null && y + 1 < n && matrix[x][y + 1] == -1) {
				matrix[x][y++] = head.val;
				head = head.next;
			}
			while (head.next != null && x + 1 < m && matrix[x + 1][y] == -1) {
				matrix[x++][y] = head.val;
				head = head.next;
			}
			while (head.next != null && y > 0 && matrix[x][y - 1] == -1) {
				matrix[x][y--] = head.val;
				head = head.next;
			}
			while (head.next != null && x > 0 && matrix[x - 1][y] == -1) {
				matrix[x--][y] = head.val;
				head = head.next;
			}
		}
		matrix[x][y] = head.val;
		return matrix;
	}
}
