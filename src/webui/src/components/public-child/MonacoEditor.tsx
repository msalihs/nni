import * as React from 'react';
import { Spinner } from 'office-ui-fabric-react';
import { DRAWEROPTION } from '../../static/const';
import MonacoEditor from 'react-monaco-editor';

interface MonacoEditorProps {
    content: string;
    loading: boolean;
    height: number;
}

class MonacoHTML extends React.Component<MonacoEditorProps, {}> {

    public _isMonacoMount!: boolean;

    constructor(props: MonacoEditorProps) {
        super(props);
    }

    componentDidMount(): void {
        this._isMonacoMount = true;
    }

    componentWillUnmount(): void {
        this._isMonacoMount = false;
    }

    render(): React.ReactNode {
        const { content, loading, height } = this.props;
        return (
            <div className="just-for-log">
                {
                    loading
                        ?
                        <Spinner
                            label="Wait, wait..."
                            ariaLive="assertive"
                            labelPosition="right"
                            styles={{ root: { width: '100%', height: height } }}
                        >
                            <MonacoEditor
                                width="100%"
                                height={height}
                                language="json"
                                value={content}
                                options={DRAWEROPTION}
                            />
                        </Spinner>
                        :
                        <MonacoEditor
                            width="100%"
                            height={height}
                            language="json"
                            value={content}
                            options={DRAWEROPTION}
                        />
                }

            </div>
        );
    }
}

export default MonacoHTML;
