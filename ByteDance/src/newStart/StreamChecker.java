package newStart;

/**
 * 字符流
 */
public class StreamChecker {

    class TrieNode {
        boolean isEnd;
        TrieNode[] tns = new TrieNode[26];
    }

    TrieNode root = new TrieNode();
    String s = "";

    public StreamChecker(String[] words) {
        for (String str : words) {
            insertReverse(str);
        }
    }

    private void insertReverse(String str) {
        TrieNode cur = root;
        for (int i = str.length() - 1; i >= 0; i--) {
            int idx = str.charAt(i) - 'a';
            if (cur.tns[idx] == null) {
                cur.tns[idx] = new TrieNode();
            }
            cur = cur.tns[idx];
        }
        cur.isEnd = true;
    }

    public boolean query(char letter) {
        TrieNode cur = root;
        s = s + letter;
        int end = Math.max(0,s.length() - 200);
        for (int i = s.length() - 1;i >= end;i--){
            int idx = s.charAt(i) - 'a';
            if (cur.tns[idx] == null){
                return false;
            }else if (cur.tns[idx].isEnd){
                return true;
            }
            cur = cur.tns[idx];
        }
        return cur.isEnd;
    }
}
