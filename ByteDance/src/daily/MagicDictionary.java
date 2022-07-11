package daily;

/**
 * @author Kezi
 * @date 2022年07月11日 22:51
 * 实现一个魔法字典
 */
public class MagicDictionary {
	String[] ss;

	public void buildDict(String[] _ss) {
		ss = _ss;
	}

	public boolean search(String str) {
		for (String s : ss) {
			int cnt = 0;
			for (int i = 0; s.length() == str.length() && i < s.length() && cnt <= 1; i++) {
				if (s.charAt(i) != str.charAt(i)) cnt++;
			}
			if (cnt == 1) return true;
		}
		return false;
	}
}
