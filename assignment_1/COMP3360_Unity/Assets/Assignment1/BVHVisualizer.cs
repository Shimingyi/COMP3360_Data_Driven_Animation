#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;
using System.IO;
using System.Collections.Generic;

[ExecuteInEditMode]
public class BVHVisualizer : MonoBehaviour
{
    public static float CMtoM = 0.01f;
    // Motion Data
    public string bvhFile = "RunRandom"; 
    public GameObject characterRoot;
    public int targetFrameRate = 60;
    public MotionData currentData = new MotionData(0);

    // Motion Player Control
    private bool Playing = false;
    private int currentFrame = 0;
    
    public static MotionData LoadBVH(string bvhFile){
        // todo 1. BVH Files Should Be Placed Inside "/Assignment1/BVHs/" Folder
        // todo 1. End
        // Check Full Path
        string fullPath = Application.dataPath + "/Assignment1/BVHs/" + bvhFile + ".bvh";
        if(!File.Exists(fullPath)){
            Debug.LogError("file " + fullPath + "not exists.");
            return null;
        }

        // Read All Lines
        string[] lines = System.IO.File.ReadAllLines(fullPath);
        char[] whitespace = new char[] {' '};
        int index = 0;

        // Create Source Data
        Character character = new Character();
        List<Vector3> offsets = new List<Vector3>();
        List<int[]> channels = new List<int[]>();
        List<float[]> motions = new List<float[]>();
        string name = string.Empty;
        string parent = string.Empty;
        Vector3 offset = Vector3.zero;
        int[] channel = null;

        // Create Hierachy From BVH
        for(index = 0; index<lines.Length; index++) {
            if(lines[index] == "MOTION") {
                break;
            }
            string[] entries = lines[index].Split(whitespace);
            for(int entry=0; entry<entries.Length; entry++) {
                if(entries[entry].Contains("ROOT")) {
                    parent = "None";
                    name = entries[entry+1];
                    break;
                } else if(entries[entry].Contains("JOINT")) {
                    parent = name;
                    name = entries[entry+1];
                    break;
                } else if(entries[entry].Contains("End")) {
                    parent = name;
                    name = name+entries[entry+1];
                    string[] subEntries = lines[index+2].Split(whitespace);
                    for(int subEntry=0; subEntry<subEntries.Length; subEntry++) {
                        if(subEntries[subEntry].Contains("OFFSET")) {
                            offset.x = FileUtility.ReadFloat(subEntries[subEntry+1]);
                            offset.y = FileUtility.ReadFloat(subEntries[subEntry+2]);
                            offset.z = FileUtility.ReadFloat(subEntries[subEntry+3]);
                            break;
                        }
                    }
                    character.AddBone(name, parent);
                    offsets.Add(offset);
                    channels.Add(new int[0]);
                    index += 2;
                    break;
                } else if(entries[entry].Contains("OFFSET")) {
                    offset.x = FileUtility.ReadFloat(entries[entry+1]);
                    offset.y = FileUtility.ReadFloat(entries[entry+2]);
                    offset.z = FileUtility.ReadFloat(entries[entry+3]);
                    break;
                } else if(entries[entry].Contains("CHANNELS")) {
                    // Use Channel To Record Paremeter Type
                    channel = new int[FileUtility.ReadInt(entries[entry+1])];
                    for(int i=0; i<channel.Length; i++) {
                        if(entries[entry+2+i] == "Xposition") {
                            channel[i] = 1;
                        } else if(entries[entry+2+i] == "Yposition") {
                            channel[i] = 2;
                        } else if(entries[entry+2+i] == "Zposition") {
                            channel[i] = 3;
                        } else if(entries[entry+2+i] == "Xrotation") {
                            channel[i] = 4;
                        } else if(entries[entry+2+i] == "Yrotation") {
                            channel[i] = 5;
                        } else if(entries[entry+2+i] == "Zrotation") {
                            channel[i] = 6;
                        }
                    }
                    character.AddBone(name, parent);
                    offsets.Add(offset);
                    channels.Add(channel);
                    break;
                } else if(entries[entry].Contains("}")) {
                    name = parent;
                    parent = name == "None" ? "None" : character.FindBone(name).parent;
                    break;
                }
            }
        }
        index += 1;

        // Remove Empty Lines
        while(lines[index].Length == 0) { 
            index += 1;
        }
        // Create Motion Data By Reading Frame Number
        MotionData motionData = new MotionData(FileUtility.ReadInt(lines[index].Substring(8)));
        // Save Character
        motionData.character = character;
        // Set Framerate
        index += 1;
        motionData.frameRate = Mathf.RoundToInt(1f / FileUtility.ReadFloat(lines[index].Substring(12)));

        // Read Motion Data
        index += 1;
        for(int i=index; i<lines.Length; i++) {
            motions.Add(FileUtility.ReadArray(lines[i]));
        }

        // Save Motion Data As Poses
        for(int k=0; k<motionData.frameNum; k++) {
            motionData.poses[k] = new Pose(character.bones.Length);
            int idx = 0;
            for(int i=0; i<character.bones.Length; i++) {
                Vector3 position = Vector3.zero;
                Quaternion rotation = Quaternion.identity;
                //todo 2. Create Local Position and Rotation From "motions" and Channel Information "channels"
                for(int j=0; j<channels[i].Length; j++) {
                    if(channels[i][j] == 1) {
                        position.x = motions[k][idx]; idx += 1;
                    }
                    if(channels[i][j] == 2) {
                        position.y = motions[k][idx]; idx += 1;
                    }
                    if(channels[i][j] == 3) {
                        position.z = motions[k][idx]; idx += 1;
                    }
                    if(channels[i][j] == 4) {
                        rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.right); idx += 1;
                    }
                    if(channels[i][j] == 5) {
                        rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.up); idx += 1;
                    }
                    if(channels[i][j] == 6) {
                        rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.forward); idx += 1;
                    }
                }
                // todo 2. End
                // Set Position As Default Offset (Initial Pose) If Only Contain Rotation
                position = (position == Vector3.zero ? offsets[i] : position);
                // Save As Local Position/Rotation, CMtoM - Real Character Is Built In Meter Scale
                motionData.poses[k].localPositions[i] = position * CMtoM;
                motionData.poses[k].localRotations[i] = rotation;
            }
        }
        Debug.Log(fullPath + "is loaded");
        return motionData;
    }

    [System.Serializable]
    public class MotionData{
        public int frameNum; 
        public int frameRate;
        public Character character;
        public Pose[] poses;
        public MotionData(int _frameNum){
            if(frameNum>=0){
                frameNum = _frameNum;
                poses = new Pose[frameNum];
            }
        }
    }
    [System.Serializable]
    public class Pose {
        public int jointNum; 
        public Vector3[] localPositions;
        public Quaternion[] localRotations;
        public Pose(int _jointNum){
            if(_jointNum>0){
                jointNum = _jointNum;
                localPositions = new Vector3[_jointNum];
                localRotations = new Quaternion[_jointNum];
            }
        }
	}

    [System.Serializable]  
	public class Character {
		public Bone[] bones;
		private string[] names = null;
		public Character() {
			bones = new Bone[0];
		}
		public void AddBone(string name, string parent) {
			ArrayExtensions.Add(ref bones, new Bone(bones.Length, name, parent));
		}
		public Bone FindBone(string name) {
			return System.Array.Find(bones, x => x.name == name);
		}
		public string[] GetBoneNames() {
			if(names == null || names.Length != bones.Length) {
				names = new string[bones.Length];
				for(int i=0; i<bones.Length; i++) {
					names[i] = bones[i].name;
				}
			}
			return names;
		}

        [System.Serializable]  
		public class Bone {
			public int index = -1;
			public string name = "";
			public string parent = "";
            public Transform trans = null;
			public Bone(int _index, string _name, string _parent) {
				index = _index;
				name = _name;
				parent = _parent;
			}
		}
	}
    
    // Bind Real Character (GameObject/Transform) To Motion Data
    public void BindCharacter(){
        if(characterRoot == null){
            characterRoot = GameObject.Find("Root");
            if(characterRoot == null){
                Debug.LogError("Can not find character root");
                return;
            }
        }
        Transform[] transforms = characterRoot.GetComponentsInChildren<Transform>();
        for(int i=1; i<transforms.Length; i++){
            Character.Bone bone =  currentData.character.FindBone(transforms[i].name);
            if(bone!=null){
                bone.trans = transforms[i];
            }
            else{
                bone.trans = null;
            }
        }
    }


    // Simple FK: Assign Pose To Character 
    private void LoadFrame(int frameIndex){
        // todo 3. Simple FK: Assign Pose to Character
        for(int i=0; i<currentData.character.bones.Length; i++){
            if(i==0){
                // If Root, Then Apply Rotation And Position
                currentData.character.bones[i].trans.localPosition = currentData.poses[frameIndex].localPositions[i];
                currentData.character.bones[i].trans.localRotation = currentData.poses[frameIndex].localRotations[i];
            }
            if(currentData.character.bones[i].trans!=null){
                // If Not Root, Can Ignore localPosition Here 
                // currentData.character.bones[i].trans.localPosition = currentData.poses[frameIndex].localPositions[i];
                currentData.character.bones[i].trans.localRotation = currentData.poses[frameIndex].localRotations[i];
            }
        }
        // todo 3. End
    }

    // Control The Motion Playing
    private IEnumerator Play() {
        System.DateTime previous = Utility.GetTimestamp();
        while(currentData.frameNum>0){
            while(Utility.GetElapsedTime(previous) < 1f/targetFrameRate) {
                yield return new WaitForSeconds(0f);
            }
            System.DateTime current = Utility.GetTimestamp();
            currentFrame = currentFrame >= currentData.frameNum-1? 0 : currentFrame+ (currentData.frameRate/targetFrameRate);
            {
                LoadFrame(currentFrame);
                // Debug.Log(currentFrame);
            }
            previous = current;
        }
	}
    // Control The Motion Playing
    public void PlayAnimation() {
        if(Playing) {
			return;
		}
		Playing = true;
		EditorCoroutines.StartCoroutine(Play(), this);
	}
    // Control The Motion Playing
    public void StopAnimation() {
        if(!Playing) {
			return;
		}
		Playing = false;
		EditorCoroutines.StopCoroutine(Play(), this);
	}


    // Editor
    [CustomEditor(typeof(BVHVisualizer))]
	public class Visualizer_Editor : Editor {
        private BVHVisualizer bvhVisualizer;
        private System.DateTime Timestamp;
        void Awake() {
            bvhVisualizer = (BVHVisualizer)target;
            Timestamp = Utility.GetTimestamp();
            EditorApplication.update += EditorUpdate;
        }
        void OnDestroy() {
			EditorApplication.update -= EditorUpdate;
		}
        public override void OnInspectorGUI() {
			Inspector();
			if(GUI.changed) {
				EditorUtility.SetDirty(bvhVisualizer);
			}
		}
        public void EditorUpdate() {
			if(Utility.GetElapsedTime(Timestamp) >= 1f/10f) {
				Repaint();
				Timestamp = Utility.GetTimestamp();
			}
		}
        // Editor
        public void Inspector() {
            DrawDefaultInspector();
            if(Utility.GUIButton("Load", UltiDraw.DarkGrey, UltiDraw.White)) {
                // When Press Load Button
                // LoadBVH -> BindCharacter -> Init Frame Number
                bvhVisualizer.currentData = BVHVisualizer.LoadBVH(bvhVisualizer.bvhFile);
                bvhVisualizer.BindCharacter();
                bvhVisualizer.currentFrame = 0;
                bvhVisualizer.LoadFrame(bvhVisualizer.currentFrame);
            }
            if(bvhVisualizer.currentData.frameNum>0){
                if(bvhVisualizer.Playing) {
                    if(Utility.GUIButton("||", Color.red, Color.black, 50f, 20f)) {
                        bvhVisualizer.StopAnimation();
                    }
                } else {
                    if(Utility.GUIButton("|>", Color.green, Color.black, 50f, 20f)) {
                        bvhVisualizer.PlayAnimation();
                    }
                }
                int index = EditorGUILayout.IntSlider(bvhVisualizer.currentFrame, 0, bvhVisualizer.currentData.frameNum-1);
                if(index != bvhVisualizer.currentFrame) {
                    bvhVisualizer.currentFrame = index;
                    bvhVisualizer.LoadFrame(bvhVisualizer.currentFrame);
                }
            }
            else
            {
                EditorGUILayout.TextField("Please load the motions");
            }
        }
    }
}
#endif